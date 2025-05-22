import hydra, torch, numpy as np, random, os, logging
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from Network.base_go_classifier import BaseGOClassifier
from metrics import MetricBundle
from Network.model import InterlabelGODataset   # your existing dataset
from Network.model_utils import EarlyStop      # keep early-stop helper
from pathlib import Path
from tqdm import tqdm
from Network.base_go_classifier import BaseGOClassifier
 
def setup_logger(log_dir: str):
    logger = logging.getLogger("trainer")
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # File handler logs everything
    fh = logging.FileHandler(os.path.join(log_dir, "train.log"))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)  
    logger.addHandler(fh)

    # Console handler logs only INFO and above
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger

@hydra.main(version_base=None, config_path="configs", config_name="bpo")
def main(cfg: DictConfig):
    # reproducibility ----------
    random.seed(cfg.seed); np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed); torch.cuda.manual_seed_all(cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # setup logger
    logger = setup_logger(cfg.log.out_dir)

    # data ----------------------
    train_ds = InterlabelGODataset(
        features_dir=cfg.data_dir,
        embedding_type=cfg.dataset.embedding_type,
        names_npy=cfg.dataset.train_names,
        labels_npy=cfg.dataset.train_labels,
        # repr_layers=cfg.dataset.repr_layers,
        # combine_feature=True,
    )
    valid_ds = InterlabelGODataset(
        features_dir=cfg.data_dir,
        embedding_type=cfg.dataset.embedding_type,
        names_npy=cfg.dataset.valid_names,
        labels_npy=cfg.dataset.valid_labels,
        # repr_layers=cfg.dataset.repr_layers,
        # combine_feature=True,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size   = cfg.dataset.batch_size,
        shuffle      = True,
        num_workers  = cfg.dataset.num_workers,
        drop_last    = True          # avoid 1‑sample batch → BatchNorm error
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=cfg.dataset.batch_size,
        shuffle=False, num_workers=cfg.dataset.num_workers
    )

    # infer embedding dimension once
    sample = next(iter(train_loader))[1]           # (names, feats, labels)
    input_dim = sample.shape[1]
    cfg.model.input_dim = input_dim


    # model / optim -------------
    if cfg.dataset.embedding_type == "mmsite":
        from Network.dnn import AP_align_fuse
        # instantiate the mmstie fusion model
        tau = getattr(cfg.model, "tau", 0.8)
        model = AP_align_fuse(tau, hidden_size=256).to(device)
        # load pretrained fusion checkpoint and drop its old classifier head
        ckpt = cfg.model.get(
            "mmstie_pretrained",
            "/SAN/bioinf/PFP/pretrained/best_model_fuse_0.8322829131652661.pt"
        )
        old_weights = torch.load(ckpt, map_location=device)
        old_weights.pop("classifier_token.weight", None)
        old_weights.pop("classifier_token.bias",   None)
        missing_keys, unexpected_keys = model.load_state_dict(old_weights, strict=False)
        logger.info(f"MMSTIE pretrained loaded; missing keys: {missing_keys}")

        # ------------------------------------------------------------------
        # Replace the pretrained classifier with a fresh one that matches the
        # current ontology's output_dim (e.g. 453 for CCO, 673 for MFO).
        # ------------------------------------------------------------------

        hidden_size   = model.fc2_res.out_features  # 256
        new_out_dim   = int(cfg.model.output_dim)
        model.classifier = BaseGOClassifier(
            input_dim=hidden_size,
            output_dim=new_out_dim,
            projection_dim=hidden_size,
            hidden_dim=hidden_size
        ).to(device)

        # Make sure only the classifier (and any selectively unfrozen modules)
        # require gradients
        for p in model.parameters():
            p.requires_grad = False
        for p in model.classifier.parameters():
            p.requires_grad = True
        # for p in model.fc2_res.parameters():
        #     p.requires_grad = True
        # for p in model.bn2_res.parameters():
        #     p.requires_grad = True
        # for p in model.fusion_cross.parameters():
        #     p.requires_grad = True


    else:
        model = BaseGOClassifier(**cfg.model).to(device)

    # -------------- detailed run information -----------------
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Embedding type : {cfg.dataset.embedding_type}")
    logger.info(f"Input dim      : {cfg.model.input_dim}")
    logger.info(f"Output dim     : {cfg.model.output_dim}")
    logger.info(f"Total params   : {total_params:,}")
    logger.info(f"Train samples  : {len(train_ds)} | "
                f"Val samples: {len(valid_ds)} | "
                f"Batch size: {cfg.dataset.batch_size}")
    # verbose architecture to file only
    logger.debug('Model architecture:\n' + repr(model))
    
    # Only include parameters that require gradients (i.e., the classifier head for mmsite)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(
        trainable_params,
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay
    )
    criterion = torch.nn.BCEWithLogitsLoss()
    metrics   = MetricBundle(device)
    metrics_cpu = MetricBundle("cpu")        # inexpensive; avoids GPU blow‑up

    early_stop = EarlyStop(
        patience   = cfg.optim.patience,
        min_epochs = cfg.optim.get("min_epochs", 5),
        monitor    = cfg.optim.get("monitor", "loss")
    )
                
    # train loop ----------------
    for epoch in range(1, cfg.optim.epochs + 1):
        model.train(); train_loss = 0.
        if cfg.dataset.embedding_type == "mmsite":
            for _, seq_emb, text_emb, labels in tqdm(train_loader,
                                                     desc=f"Epoch {epoch:02d} [train]",
                                                     leave=False):
                seq_emb  = seq_emb.to(device)
                text_emb = text_emb.to(device)
                labels   = labels.to(device).float()
                opt.zero_grad()
                outputs = model(text_emb, seq_emb)
                logits  = outputs["token_logits"]
                loss    = criterion(logits, labels)
                # include contrastive loss if present
                # if "contrastive_loss" in outputs:
                #     loss = loss + cfg.model.get("contrastive_lambda", 5e-3) * outputs["contrastive_loss"]
                loss.backward()
                opt.step()
                train_loss += loss.item() * labels.size(0)
        else:
            for _, feats, labels in tqdm(train_loader,
                                         desc=f"Epoch {epoch:02d} [train]",
                                         leave=False):
                feats, labels = feats.to(device), labels.to(device).float()
                opt.zero_grad()
                logits = model(feats)
                loss = criterion(logits, labels)
                loss.backward(); opt.step()
                train_loss += loss.item() * feats.size(0)

        train_loss /= len(train_loader.dataset)

        # ---- validation ----
        if epoch % cfg.log.metrics_every == 0:
            model.eval(); val_loss = 0.
            all_logits, all_labels = [], []
            with torch.no_grad():
                if cfg.dataset.embedding_type == "mmsite":
                    for _, seq_emb, text_emb, labels in tqdm(valid_loader,
                                                             desc=f"Epoch {epoch:02d} [val] ",
                                                             leave=False):
                        seq_emb  = seq_emb.to(device)
                        text_emb = text_emb.to(device)
                        labels   = labels.to(device).float()
                        outputs  = model(text_emb, seq_emb)
                        logits   = outputs["token_logits"]
                        val_loss += criterion(logits, labels).item() * labels.size(0)
                        all_logits.append(logits.detach().cpu())
                        all_labels.append(labels.detach().cpu())
                else:
                    for _, feats, labels in tqdm(valid_loader,
                                                 desc=f"Epoch {epoch:02d} [val] ",
                                                 leave=False):
                        feats, labels = feats.to(device), labels.to(device).float()
                        logits = model(feats)
                        val_loss += criterion(logits, labels).item() * feats.size(0)
                        all_logits.append(logits.detach().cpu())
                        all_labels.append(labels.detach().cpu())
            val_loss /= len(valid_loader.dataset)
            metrics_report = metrics_cpu(
                torch.cat(all_logits, dim=0),
                torch.cat(all_labels, dim=0)
            )

            logger.debug(f"Epoch {epoch:02d} | "
                  f"train {train_loss:.4f} | "
                  f"val {val_loss:.4f} | "
                  f"mAP {metrics_report['AP']:.4f} | "
                  f"Fmax {metrics_report['Fmax']:.4f}")

            # early stopping: pass val_loss, current F-max, and model
            early_stop(val_loss, metrics_report["Fmax"], model)

            if early_stop.stop():                                 # query the flag
                break

    # restore best model weights (by chosen monitor) and save
    if early_stop.has_backup_model():
        model = early_stop.restore(model)
        ckpt_dir = Path(cfg.log.out_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        best_path = ckpt_dir / "best.pt"
        torch.save(model.state_dict(), best_path)
        logger.debug(f"Best checkpoint saved to: {best_path}")
    logger.debug(f"Best checkpoint object: {best_path}")
    
if __name__ == "__main__":
    main()