import os
from Bio import SeqIO
import argparse
from tqdm import tqdm
import torch
from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained
# from esm import pretrained
# from esm.data import  FastaBatchedDataset
# from esm import pretrained
# from esm import pretrained
import esm
import numpy as np
import re
import json
from transformers import T5Tokenizer, T5EncoderModel
import time
import ankh
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
from huggingface_hub import login
from settings import settings_dict as settings
from transformers import AutoModel, AutoTokenizer

from argparse import Namespace
torch.serialization.add_safe_globals([Namespace])

class PlmEmbed:

    def __init__(self,
        fasta_file:str,
        working_dir:str,
        model_name:str="esm2_t36_3B_UR50D",
        model_path:str=settings['esm3b_path'],
        use_gpu:bool=True,
        repr_layers:list=[34, 35, 36],
        include:list=["mean"],
        cache_dir:str=None,

    ):

        
        working_dir = os.path.abspath(working_dir)





        if not os.path.exists(working_dir):
            os.makedirs(working_dir)

        if cache_dir is None:
            cache_dir = os.path.join(working_dir, "embed_feature")
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        self.cache_dir = cache_dir

        self.use_gpu = use_gpu
        self.fasta_file = fasta_file
        # fasta for esm only contains the proteins that are not in the esm feature cache
        self.filltered_fasta_file = os.path.join(working_dir, f'filtered.fasta')
        self.repr_layers = repr_layers
        # note, if model_path is not provided, the model will be loaded from the path, however, the contact-regression should be put in same directory as the model_path
        self.model_path = model_path
        self.model_name = model_name
        self.include = include



    def parse_fasta(self, fasta_file=None)->dict:
        '''
        parse fasta file

        args:
            fasta_file: fasta file path
        return:
            fasta_dict: fasta dictionary {id: sequence}
        '''
        if fasta_file is None:
            fasta_file = self.fasta_file

        fasta_dict = {}
        for record in SeqIO.parse(fasta_file, 'fasta'):
            fasta_dict[record.id] = str(record.seq)
        return fasta_dict  
    
    def filter_fasta(self, fasta_file=None, cache_dir=None, filltered_fasta_file=None):
        """
        Only keep fasta sequences that are not in feature store directory

        Args:
            fasta_file (str, optional): fasta file path. Defaults to None.
        """


        print(fasta_file, cache_dir, filltered_fasta_file)
        
        if fasta_file is None:
            fasta_file = self.fasta_file
        if cache_dir is None: # place store esm embedding
            cache_dir = self.cache_dir
        if filltered_fasta_file is None:
            filltered_fasta_file = self.filltered_fasta_file




        fasta_dict = self.parse_fasta(fasta_file)
        # get processed fasta ids
        processed_ids = set()

        for file in os.listdir(cache_dir):
            if file.endswith('.npy'):
                processed_ids.add(file.split('.')[0])

        # filter fasta file
        filltered_fasta_dict = {k:v for k,v in fasta_dict.items() if k not in processed_ids}

        # write filtered fasta file
        with open(filltered_fasta_file, 'w') as f:
            for k,v in filltered_fasta_dict.items():
                f.write(f'>{k}\n{v}\n')



        return filltered_fasta_dict, filltered_fasta_file
    
    def extract(
            self,
            fasta_file:str=None, # path of fasta file to extract features from
            repr_layers:list = [34,35,36], # which layers to extract features from
            model_path:str = None, # path to model
            model_name:str = None, # name of model, if model_path is not provided
            use_gpu:bool = True, # use GPU if available
            truncate:bool = True, # truncate sequences longer than 1024 to match training setup
            include:list = ["mean", "per_tok", "bos", "contacts"], # which representations to return
            batch_size:int = 4096, # maximum batch size
            output_dir:str = None, # output directory for extracted representations
            overwrite:bool = False, # overwrite existing files
            model_type:str = "esm", # model type, esm or t5
            ) -> None:

        if model_type == "esm":
            self.esm_extract(
                fasta_file=fasta_file,
                repr_layers=repr_layers,
                model_path=model_path,
                model_name=model_name,
                use_gpu=use_gpu,
                truncate=truncate,
                include=include,
                batch_size=batch_size,
                output_dir=output_dir,
                overwrite=overwrite,
            )






        if model_type == "residue":
            self.esm_residue(
                fasta_file=fasta_file,
                # repr_layers=[34],
                # model_path=model_path,
                # model_name=model_name,
                use_gpu=use_gpu,
                # truncate=truncate,
                # include=["per_tok"],
                batch_size=batch_size,
                output_dir=output_dir,
                overwrite=overwrite,
            )
        elif model_type == "t5":

            self.t5_extract(
                fasta_file=fasta_file,
                repr_layers=repr_layers,
                model_path=model_path,
                model_name=model_name,
                use_gpu=use_gpu,
                truncate=truncate,
                include=include,
                batch_size=batch_size,
                output_dir=output_dir,
                overwrite=overwrite,
            )
        elif model_type == "ankh":
             self.ankh_extract(
                fasta_file=fasta_file,
                repr_layers=repr_layers,
                model_path=model_path,
                model_name=model_name,
                use_gpu=use_gpu,
                truncate=truncate,
                include=include,
                batch_size=batch_size,
                output_dir=output_dir,
                overwrite=overwrite,
            )
        elif model_type == "esmc":
            self.esmc_extract(
                fasta_file=fasta_file,
                model_path=model_path,
                model_name=model_name,
                use_gpu=use_gpu,
                truncate=truncate,
                include=include,
                batch_size=batch_size,
                output_dir=output_dir,
                overwrite=overwrite,
            )

        elif model_type == "text":
            self.text_extract(
                fasta_file=fasta_file,
                model_path=model_path,
                model_name=model_name,

                # batch_size=batch_size,
                output_dir=output_dir,
                overwrite=overwrite,
            )
        else:
            raise NotImplementedError(f"model type {model_type} is not implemented")
    


    def text_extract(
            self,
            output_dir,
            fasta_file,
            model_name,
            model_path,
            overwrite:bool = False,
    ):
        if output_dir is None:
            output_dir = self.cache_dir
        if fasta_file is None:
            fasta_file = self.fasta_file     
        if model_name is None:
            model_name = self.model_name
        if model_path is None:
            model_path = self.model_path
        output_dir = '/Users/zjzhou/Downloads/InterLabelGO/testdata/text'

        if not overwrite:
            print("Check file exist")
            _, fasta_file = self.filter_fasta(fasta_file, output_dir)

        if os.path.getsize(fasta_file) == 0:
            print(f'### {self.extract.__name__} ###')
            print(f'All sequences are already processed in the default cache directory. Return.')
            return
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)


        prior_knowledge = json.load(open('/Users/zjzhou/Downloads/InterLabelGO/testdata/generated_desc.json', 'r'))

        model_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"  
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)

        # Move model to GPU if available (or MPS on macOS)
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        model.to(device)
        model.eval()  # we only need inference mode

        # ------------------------------------------------------------
        # 3) For each protein's text, tokenize & compute the embeddings
        # ------------------------------------------------------------


        protein_embeddings = {}
        max_length = 128  # Your desired max length for text inputs
        with torch.no_grad():
            for protein_id, text_str in prior_knowledge.items():
                # 3A) Tokenize
                encoded_input = tokenizer(
                    text_str,
                    max_length=max_length,
                    padding="max_length",
                    truncation  =True,
                    return_tensors="pt"
                )
                # Move token tensors to device
                input_ids = encoded_input["input_ids"].to(device)
                attention_mask = encoded_input["attention_mask"].to(device)

                # 3B) Forward pass
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                outputs = outputs[0]
                # outputs.last_hidden_state.shape == [batch_size=1, seq_len=128, hidden_dim]

                # 3C) You can pick your preferred pooling strategy:
                # e.g. *average pooling* across sequence length
                # embedding = outputs.last_hidden_state.mean(dim=1)  # => shape [1, hidden_dim]

                embedding = outputs.squeeze(dim=0)               # => shape [hidden_dim]

                # 3D) Convert to numpy and store in a dict
                embedding_np = embedding.cpu().numpy()
                protein_embeddings[protein_id] = embedding_np

                # Optionally, save each embedding directly to a .npy
                out_path = os.path.join(output_dir, f"{protein_id}.npy")
                np.save(out_path, embedding_np, allow_pickle=False)
    



    def esm_extract(
            self,
            fasta_file:str=None, # path of fasta file to extract features from
            repr_layers:list = [34,35,36], # which layers to extract features from
            model_path:str = None, # path to model
            model_name:str = None, # name of model, if model_path is not provided
            use_gpu:bool = True, # use GPU if available
            truncate:bool = True, # truncate sequences longer than 1024 to match training setup
            include:list = ["mean", "per_tok", "bos", "contacts"], # which representations to return
            batch_size:int = 4096, # maximum batch size
            output_dir:str = None, # output directory for extracted representations
            overwrite:bool = False, # overwrite existing files
            ) -> None:


        if output_dir is None:
            output_dir = self.cache_dir
        if fasta_file is None:
            fasta_file = self.fasta_file     
        if model_name is None:
            model_name = self.model_name
        if model_path is None:
            model_path = self.model_path
        output_dir = '/Users/zjzhou/Downloads/InterLabelGO/testdata/embedding'



        
        # exit(fasta_file)
        # filter fasta file
        if not overwrite:
            print("Check file exist")
            _, fasta_file = self.filter_fasta(fasta_file, output_dir)
            # print(fasta_file)
            # exit()




        # Doing new embedding here
        # if os.path.getsize(fasta_file1) != 0:
        #     output_dir = "./new_embedding"
        #     if not os.path.exists(output_dir):
        #         os.makedirs(output_dir)


        if os.path.getsize(fasta_file) == 0:
            
            print(f'### {self.extract.__name__} ###')
            print(f'All sequences are already processed in the default cache directory. Return.')
            return
        
 

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        


        
        if model_path and os.path.exists(model_path):
            model, alphabet = pretrained.load_model_and_alphabet(model_path)
        elif model_name:
            if model_name == "esm2_t33_650M_UR50D":
                model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            elif model_name == "esm2_t36_3B_UR50D":
                model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
            else:
                raise NotImplementedError(f"model {model_name} is not implemented")
        else:
            raise ValueError("model_path or model_name must be provided")
        model.eval()
        # print(model)



        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        # device = torch.device("cpu")
        model = model.to(device)
        print('Device for embedding:', device)
        
        # if torch.cuda.is_available() and use_gpu:
        #     os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        #     model = model.cuda()
        #     print("Transferred model to GPU")
        # else:
        #     print("Using CPU")
        
        dataset = FastaBatchedDataset.from_file(fasta_file)
        batches = dataset.get_batch_indices(batch_size, extra_toks_per_seq=1)


        data_loader = torch.utils.data.DataLoader(
            dataset, collate_fn=alphabet.get_batch_converter(), batch_sampler=batches
        )
        print("1111111111111")
        assert all(-(model.num_layers + 1) <= i <= model.num_layers for i in repr_layers)
        repr_layers = [(i + model.num_layers + 1) % (model.num_layers + 1) for i in repr_layers]

        with torch.no_grad():
            for batch_idx, (lables, strs, toks) in tqdm(enumerate(data_loader), total=len(batches), desc="Extracting esm features", ascii=' >='):
                print(
                    f"Processing {batch_idx + 1} of {len(batches)} batches ({toks.size(0)} sequences)"
                )
                # if torch.cuda.is_available() and use_gpu:
                #     toks = toks.to(device="cuda", non_blocking=True)
                toks = toks.to(device, non_blocking=True )
                # The model is trained on truncated sequences and passing longer ones in at
                # infernce will cause an error. See https://github.com/facebookresearch/esm/issues/21
                if truncate:
                    toks = toks[:, :1022]

                out = model(toks, repr_layers=repr_layers, return_contacts="contacts" in include)
                #logits = out["logits"].to(device="cpu")
                representations = {
                    layer: t.to(device="cpu") for layer, t in out["representations"].items()
                }


                if "contacts" in include:
                    contacts = out["contacts"].to(device="cpu")


                for i, label in enumerate(lables):

                    result = {"name": label}

                    if "per_tok" in include:
                        result["per_tok"] = {
                            layer: t[i, 1: len(strs[i]) + 1].clone().numpy()
                            for layer, t in representations.items()
                        }
                    
                    if "mean" in include:
                        result["mean"] = {
                        layer: t[i, 1: len(strs[i]) + 1].mean(0).clone().numpy()
                            for layer, t in representations.items()
                        }

                    if "bos" in include:
                        result["bos"] = {
                            layer: t[i, 0].clone() for layer, t in representations.items()
                        }

                    if "contacts" in include:
                        result["contacts"] = contacts[i, : len(strs[i]), : len(strs[i])].clone().numpy()
                    
                    if "sum" in include:
                        result["sum"] = {
                        layer: t[i, 1: len(strs[i]) + 1].sum(0).clone().numpy()
                            for layer, t in representations.items()
                        }
                    
                    if "max" in include:
                        result["max"] = {
                        layer: t[i, 1: len(strs[i]) + 1].max(0).values.cpu().numpy()
                            for layer, t in representations.items()
                        }
                    
                    if "min" in include:
                        result["min"] = {
                        layer: t[i, 1: len(strs[i]) + 1].min(0).values.cpu().numpy()
                            for layer, t in representations.items()
                        }

                    out_file = os.path.join(output_dir, f"{label}.npy")
                    # print("embedding finishe and " , out_file, len(result) )
                    np.save(out_file, result, allow_pickle=True)

                    if torch.cuda.is_available() and use_gpu:
                        torch.cuda.empty_cache()  

    # 自定义 Dataset

    # def create_batches(format_seqences, batch_size):
    #             keys = list(sequences.keys())
    #             formatted_sequences = [format_sequence(sequences[key]) for key in keys]
    #             for i in range(0, len(keys), batch_size):
    #                 yield keys[i:i+batch_size], formatted_sequences[i:i+batch_size]
    

    def esmc_extract(
            self,
            fasta_file:str=None, # path of fasta file to extract features from
            repr_layers:list = [34,35,36], # which layers to extract features from

            model_path:str = None, # path to model
            model_name:str = None, # name of model, if model_path is not provided

            use_gpu:bool = True, # use GPU if available
            truncate:bool = True, # truncate sequences longer than 1024 to match training setup
            include:list = ["mean", "per_tok", "bos", "contacts"], # which representations to return
            batch_size:int = 4096, # maximum batch size
            output_dir:str = None, # output directory for extracted representations
            overwrite:bool = False, # overwrite existing files
            ) -> None:


        if output_dir is None:
            output_dir = 'new_embedding/esmc'
        if fasta_file is None:
            fasta_file = self.fasta_file     
        if model_name is None:
            model_name = self.model_name
        if model_path is None:
            model_path = self.model_path
        # output_dir = 'Beprof_benchmark/ESM_embed'



        
        # exit(fasta_file)
        # filter fasta file
        if not overwrite:
            print("Check file exist")
            _, fasta_file = self.filter_fasta(fasta_file, output_dir)
            # print(fasta_file)
            # exit()




        # Doing new embedding here
        # if os.path.getsize(fasta_file1) != 0:
        #     output_dir = "./new_embedding"
        #     if not os.path.exists(output_dir):
        #         os.makedirs(output_dir)


        if os.path.getsize(fasta_file) == 0:
            
            print(f'### {self.extract.__name__} ###')
            print(f'All sequences are already processed in the default cache directory. Return.')
            return
        
 

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        

        # print(model)
        login(token="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")
        # from esm.models.esmc import ESMC
        # from esm.sdk.api import ESMProtein, LogitsConfig

        # protein = ESMProtein(sequence="AAAAAAAAAA")
        # client = ESMC.from_pretrained("esmc_300m").to("cpu") # or "cpu"
        # protein_tensor = client.encode(protein)
        # logits_output = client.logits(
        # protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
        # )

        # print( protein_tensor)
        exit()

        protein = ESMProtein(sequence="AAAAAAAAAAAA")
        client = ESMC.from_pretrained("esmc_300m")
        client.eval()
        
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print('Device for embedding:', device)

        client = client.to(device)







        protein_tensor = client.encode(protein)
        logits_output = client.logits(
        protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
        )

        sequences = self.parse_fasta(fasta_file)
        exit(sequences)

        # device = torch.device("cpu")
       
   
        
        # if torch.cuda.is_available() and use_gpu:
        #     os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        #     model = model.cuda()
        #     print("Transferred model to GPU")
        # else:
        #     print("Using CPU")
        
       

    def ankh_extract(
                self,
                fasta_file:str=None, # path of fasta file to extract features from
                repr_layers:list = [34,35,36], # which layers to extract features from
                model_path:str = None, # path to model
                model_name:str = None, # name of model, if model_path is not provided
                use_gpu:bool = True, # use GPU if available
                truncate:bool = True, # truncate sequences longer than 1024 to match training setup
                include:list = ["mean", "per_tok", "bos", "contacts"], # which representations to return
                batch_size:int = 4096, # maximum batch size
                output_dir:str = None, # output directory for extracted representations
                overwrite:bool = False, # overwrite existing files
                
                ) -> None:

            if output_dir is None:
                output_dir = self.cache_dir
            if fasta_file is None:
                fasta_file = self.fasta_file     
            if model_name is None:
                model_name = self.model_name
            if model_path is None:
                model_path = self.model_path

            if not overwrite:
                print("Check file exist")

                # find the sequence not embedding before 
                output_dir = './new_embedding/ankh'
                # output_dir = 'example/ankn_test_emb'

                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                _, fasta_file = self.filter_fasta(fasta_file, output_dir)
                print("the unembedding protein store in: ", fasta_file)


            if os.path.getsize(fasta_file) == 0:
                
                print(f'### {self.extract.__name__} ###')
                print(f'All sequences are already processed in the default cache directory. Return.')
                return
            


           

            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
            print("Using device: {} for embedding".format(device))
            # transformer_link = "Rostlab/prot_t5_xl_half_uniref50-enc"
            # print("Loading: {}".format(transformer_link))
            # model = T5EncoderModel.from_pretrained(transformer_link)
            # model.full() if device=='cpu' else model.half() # only cast to full-precision if no GPU is available
            # model = model.to(device)
            # model = model.eval()
            # tokenizer = T5Tokenizer.from_pretrained(transformer_link, do_lower_case=False )
            
            # To load large model:
            model, tokenizer = ankh.load_large_model()

            model = model.to(device)

            

             # 生成批处理的 DataLoader 数据
            def create_batches(sequences, batch_size):


                MAX_SEQ_LEN = 1024
                keys = list(sequences.keys())
                formatted_sequences = [list(value)[:MAX_SEQ_LEN] for value in sequences.values()]

                for i in range(0, len(keys), batch_size):
                    yield keys[i:i+batch_size], formatted_sequences[i:i+batch_size]






            sequences = self.parse_fasta(fasta_file)


            total_sequences = len(sequences)
            print(f"Processing {total_sequences} protein sequences...")
            # exit()

            batch_size = 6
            print(batch_size)
            print(device)

            for seq_ids, batch_sequences in tqdm(create_batches(sequences, batch_size), total=total_sequences/batch_size, desc="Generating embeddings", unit="batch"):
                
                outputs = tokenizer.batch_encode_plus(batch_sequences, 
                                    add_special_tokens=True, 
                                    padding=True, 
                                    is_split_into_words=True, 
                                    return_tensors="pt")
                
                with torch.no_grad():
                    embeddings = model(input_ids=outputs['input_ids'].to(device), attention_mask=outputs['attention_mask'].to(device))


                embeddings = embeddings['last_hidden_state'].mean(dim=1)


                # # 分词并生成输入张量
                # tokenized = tokenizer(batch_sequences, add_special_tokens=True, padding="longest", return_tensors="pt")
                # input_ids = tokenized["input_ids"].to(device)
                # attention_mask = tokenized["attention_mask"].to(device)
                # MAX_SEQ_LEN = 1024  # Define a maximum length
                # input_ids = input_ids[:, :MAX_SEQ_LEN]
                # attention_mask = attention_mask[:, :MAX_SEQ_LEN]

                # 生成嵌入


                # 保存每个序列的嵌入
                for seq_id, embedding in zip(seq_ids, embeddings):
                    output_file = os.path.join(output_dir, f"{seq_id}.npy")
                    np.save(output_file, embedding.cpu().numpy())
                    tqdm.write(f"Saved embedding for {seq_id} to {output_file}")

                

                




    

                       
    def t5_extract(
                self,
                fasta_file:str=None, # path of fasta file to extract features from
                repr_layers:list = [34,35,36], # which layers to extract features from
                model_path:str = None, # path to model
                model_name:str = None, # name of model, if model_path is not provided
                use_gpu:bool = True, # use GPU if available
                truncate:bool = True, # truncate sequences longer than 1024 to match training setup
                include:list = ["mean", "per_tok", "bos", "contacts"], # which representations to return
                batch_size:int = 4096, # maximum batch size
                output_dir:str = None, # output directory for extracted representations
                overwrite:bool = False, # overwrite existing files
                
                ) -> None:

            if output_dir is None:
                output_dir = self.cache_dir
            if fasta_file is None:
                fasta_file = self.fasta_file     
            if model_name is None:
                model_name = self.model_name
            if model_path is None:
                model_path = self.model_path

            if not overwrite:
                print("Check file exist")

                # find the sequence not embedding before 
                output_dir = './new_embedding/t5'
                output_dir = 'example/T5_test_emb'

                _, fasta_file = self.filter_fasta(fasta_file, output_dir)



            if os.path.getsize(fasta_file) == 0:
                
                print(f'### {self.extract.__name__} ###')
                print(f'All sequences are already processed in the default cache directory. Return.')
                return
            

            if not os.path.exists(output_dir):


                os.makedirs(output_dir)
           

            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
            print("Using device: {} for embedding".format(device))
            transformer_link = "Rostlab/prot_t5_xl_half_uniref50-enc"
            print("Loading: {}".format(transformer_link))
            model = T5EncoderModel.from_pretrained(transformer_link)
            model.full() if device=='cpu' else model.half() # only cast to full-precision if no GPU is available
            model = model.to(device)
            model = model.eval()
            tokenizer = T5Tokenizer.from_pretrained(transformer_link, do_lower_case=False )

            

             # 生成批处理的 DataLoader 数据
            def create_batches(sequences, batch_size):


                # print("kdjlsa;jf;dljljdsl;jdfslj")

                # print(sequences)


                keys = list(sequences.keys())

                formatted_sequences = [" ".join(list(   re.sub(r"[UZOB]", "X", sequences[key])))  for key in keys]

                for i in range(0, len(keys), batch_size):
                    yield keys[i:i+batch_size], formatted_sequences[i:i+batch_size]




            

            sequences = self.parse_fasta(fasta_file)


            total_sequences = len(sequences)
            print(f"Processing {total_sequences} protein sequences...")


            batch_size = 16
            print(batch_size)
            print(device)

            for seq_ids, batch_sequences in tqdm(create_batches(sequences, batch_size), total=total_sequences/batch_size, desc="Generating embeddings", unit="batch"):
                

                # 分词并生成输入张量
                tokenized = tokenizer(batch_sequences, add_special_tokens=True, padding="longest", return_tensors="pt")
                input_ids = tokenized["input_ids"].to(device)
                attention_mask = tokenized["attention_mask"].to(device)
                MAX_SEQ_LEN = 1024  # Define a maximum length
                input_ids = input_ids[:, :MAX_SEQ_LEN]
                attention_mask = attention_mask[:, :MAX_SEQ_LEN]

                # 生成嵌入
                with torch.no_grad():
                    embedding_repr = model(input_ids=input_ids, attention_mask=attention_mask)

                    embeddings = embedding_repr.last_hidden_state.mean(dim=1)  # 每个序列取均值


                # 保存每个序列的嵌入
                for seq_id, embedding in zip(seq_ids, embeddings):
                    output_file = os.path.join(output_dir, f"{seq_id}.npy")
                    np.save(output_file, embedding.cpu().numpy())
                    tqdm.write(f"Saved embedding for {seq_id} to {output_file}")




    

            exit()


            for seq_id, sequence in tqdm(sequences.items(), desc="Generating embeddings", unit="sequence"):

                # 替换罕见氨基酸并添加空格
                formatted_sequence = " ".join(list(re.sub(r"[UZOB]", "X", sequence)))

                # 分词并生成输入张量
                ids = tokenizer([formatted_sequence], add_special_tokens=True, padding="longest")

                input_ids = torch.tensor(ids['input_ids']).to(device)
                attention_mask = torch.tensor(ids['attention_mask']).to(device)

                # 生成嵌入
                with torch.no_grad():
                    embedding_repr = model(input_ids=input_ids, attention_mask=attention_mask)
                
                # 计算 per-protein embedding
                embedding_per_protein = embedding_repr.last_hidden_state[0].mean(dim=0)  # shape (1024,)
                
                # 保存为 .npy 文件
                output_file = os.path.join(output_dir, f"{seq_id}.npy")
                np.save(output_file, embedding_per_protein.cpu().numpy())
                tqdm.write(f"Saved embedding for {seq_id} to {output_file}")  # 可选，显示每个文件保存信息
            print("Finished here!")
            exit()
           

                        

            if torch.cuda.is_available() and use_gpu:
                    torch.cuda.empty_cache() 
                            
    # def esm_residue(
    #     self,
    #     fasta_file: str = None,
    #     model: torch.nn.Module = None,
    #     tokenizer = None,
    #     use_gpu: bool = True,
    #     batch_size: int = 32,
    #     output_dir: str = None,
    #     overwrite: bool = False,
    #     fixed_len: int = 1022
    # ) -> None:
    #     """
    #     Extract protein embeddings using the provided model in the following format:
    #     embeddings = model(input_ids=input_ids, attention_mask=attention_mask)

    #     Args:
    #         fasta_file: path to your FASTA file.
    #         model: the pre-loaded sequence embedding model.
    #         tokenizer: tokenizer corresponding to the provided model.
    #         use_gpu: whether to use GPU if available.
    #         batch_size: how many sequences per batch.
    #         output_dir: directory to save embeddings.
    #         overwrite: whether to overwrite existing embeddings.
    #         fixed_len: uniform length for padding/truncation.
    #     """
    #     # import os
    #     # import torch
    #     # from tqdm import tqdm
    #     # from esm.data import FastaBatchedDataset
    #     # import numpy as np

    #     if output_dir is None:
    #         output_dir = self.cache_dir
    #     if fasta_file is None:
    #         fasta_file = self.fasta_file

    #     if not overwrite:
    #         _, fasta_file = self.filter_fasta(fasta_file, output_dir)

    #     if os.path.getsize(fasta_file) == 0:
    #         print("All sequences are already processed. Returning.")
    #         return

    #     if not os.path.exists(output_dir):
    #         os.makedirs(output_dir)



    #     model_name = 'esm2_t33_650M_UR50D'


    #     if model_name:
    #         if model_name == "esm2_t33_650M_UR50D":
    #             model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    #         elif model_name == "esm2_t36_3B_UR50D":
    #             model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
    #         else:
    #             raise NotImplementedError(f"model {model_name} not implemented")
    #     else:
    #         raise ValueError("Either model_path or model_name must be provided.")
        
    #     device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    #     model = model.to(device)
    #     model.eval()
    #     # exit()

    #     dataset = FastaBatchedDataset.from_file(fasta_file)
    #     batches = dataset.get_batch_indices(batch_size)
    #     data_loader = torch.utils.data.DataLoader(
    #         dataset,
    #         collate_fn=lambda batch: batch,
    #         batch_sampler=batches
    #     )

    #     def pad_or_truncate(tokens, max_len, pad_token_id):
    #         if len(tokens) > max_len:
    #             return tokens[:max_len]
    #         return tokens + [pad_token_id] * (max_len - len(tokens))

    #     with torch.no_grad():
    #         for batch in tqdm(data_loader, desc="Extracting features"):
    #             labels, seq_strs = zip(*batch)

    #             encoded = tokenizer(
    #                 list(seq_strs),
    #                 padding='max_length',
    #                 truncation=True,
    #                 max_length=fixed_len,
    #                 return_tensors='pt'
    #             )

    #             input_ids = encoded['input_ids'].to(device)
    #             attention_mask = encoded['attention_mask'].to(device)

    #             embeddings = model(input_ids=input_ids, attention_mask=attention_mask)

    #             embeddings_np = embeddings.last_hidden_state.cpu().numpy()
    #             exit(embeddings_np.size())
    #             for i, label in enumerate(labels):
    #                 result = {
    #                     "name": label,
    #                     "embedding": embeddings_np[i]
    #                 }
    #                 np.save(os.path.join(output_dir, f"{label}.npy"), result)

    #     print(f"Done! Embeddings saved in {output_dir}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('fasta_file', type=str, help='path of fasta file')
    parser.add_argument('workdir', type=str, help='path of working directory')
    parser.add_argument('-mn', '--model_name', type=str, help='name of model', default="esm2_t36_3B_UR50D",)
    parser.add_argument('-mp', '--model_path', type=str, help='path of model', default="./Data/esm_models/esm2_t36_3B_UR50D.pt")
    parser.add_argument('-c', '--cache_dir', type=str, help='path of cache directory', default=None)
    parser.add_argument('--use_gpu', action='store_true', help='use gpu')
    parser.add_argument("--include", type=str, nargs="+", default=["mean"], choices=["mean", "per_tok", "bos", "contacts"], help="which representations to return")
    parser.add_argument("--repr_layers", type=int, nargs="+", default=[-3, -2, -1], help="which layers to extract features from, default is [-3, -2, -1] which means the last three layers")
    args = parser.parse_args()


    plm = PlmEmbed(
        args.fasta_file,
        args.workdir,
        model_name=args.model_name,
        model_path=args.model_path,
        cache_dir=args.cache_dir,
        use_gpu=args.use_gpu,
        include=args.include,
        repr_layers=args.repr_layers,
    )
    plm.extract(plm.fasta_file, 
    repr_layers=plm.repr_layers, 
    model_path=plm.model_path, 
    model_name=plm.model_name, 
    use_gpu=plm.use_gpu, 
    include=plm.include, 
    output_dir=plm.cache_dir,
    model_type = "t5",
    )

