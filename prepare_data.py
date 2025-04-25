from utils.obo_tools import ObOTools
import pandas as pd
import os, sys, subprocess, argparse
import numpy as np
from Bio import SeqIO
import scipy.sparse as ssp
from sklearn.model_selection import KFold

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import shutil

# from plm import PlmEmbed
from settings import settings_dict as settings

docstring = """
example usage:
    python prepare_data.py -train Data/cafa5_raw_data/train_terms.tsv -train_seqs Data/cafa5_raw_data/train_seq.fasta -d Data --make_db --ia
    python prepare_data.py -train /Users/zjzhou/Downloads/InterLabelGO/testdata/test.tsv -train_seqs /Users/zjzhou/Downloads/InterLabelGO/testdata/test.fasta -d Data --make_db --ia


    python prepare_data.py -train testdata/test.tsv -train_seqs testdata/test.fasta -d Data --make_db --ia
    grep '^>' testdata/test.fasta | sed 's/^>//' | cut -d' ' -f1 > ids.txt && grep -F -f ids.txt Data/cafa5_raw_data/train_terms.tsv > testdata/test.tsv

"""


oboTools = ObOTools()

aspects = ['BPO', 'CCO', 'MFO']

def get_seq_dict(fast_file:str)->dict:
    """read fasta file and return a dict

    Args:
        fast_file (str): path of fasta file

    Returns:
        dict: dict of fasta file, key is protein name, value is protein sequence
    """
    seq_dict = {}
    for record in SeqIO.parse(fast_file, "fasta"):
        seq_dict[record.id] = str(record.seq)
    return seq_dict

def prop_parents(df:pd.DataFrame)->pd.DataFrame:
    """propagate parent terms to the dataframe

    Args:
        df (pd.DataFrame): dataframe with columns ['EntryID', 'term']

    Returns:
        pd.DataFrame: dataframe with columns ['EntryID', 'term']
    """

    
    aspect_df = []



    for aspect in aspects:
        cur_df = df[df['aspect'] == aspect].copy()
        
        cur_df = cur_df.groupby(['EntryID'])['term'].apply(set).reset_index()


        # for each go term in a protein entry, all all it's parent terms and add back to datafrme
        cur_df['term'] = cur_df['term'].apply(lambda x: oboTools.backprop_terms(x))
        # exit()
        
        cur_df = cur_df.explode('term')

        cur_df['aspect'] = aspect
        aspect_df.append(cur_df)

    prop_df = pd.concat(aspect_df, axis=0)


    return prop_df


def make_db(Data_dir:str, terms_df:pd.DataFrame, seq_dict:dict):
    DatabaseDir = settings['alignment_db']
    goa_dir = settings['alignment_labels']

    if not os.path.exists(DatabaseDir):
        os.makedirs(DatabaseDir)
    if not os.path.exists(goa_dir):
        os.makedirs(goa_dir)

    for aspect in ['BPO', 'CCO', 'MFO']:
        aspect_db_dir = os.path.join(DatabaseDir, aspect)
        if not os.path.exists(aspect_db_dir):
            os.makedirs(aspect_db_dir)
        cur_aspect_df = terms_df[terms_df['aspect'] == aspect].copy()
        # group by EntryID, apply set to term
        cur_aspect_df = cur_aspect_df.groupby(['EntryID'])['term'].apply(set).reset_index()
        # backprop parent
        cur_aspect_df['term'] = cur_aspect_df['term'].apply(lambda x: oboTools.backprop_terms(x))
        cur_aspect_df['term'] = cur_aspect_df['term'].apply(lambda x: ','.join(x))
        unique_entryids = set(cur_aspect_df['EntryID'].unique())
        if len(unique_entryids) == 0:
            raise Exception(f"Error: no {aspect} terms found in dataframe, please make sure aspect is one of BPO, MFO, CCO")        
        # write to file
        label_path = os.path.join(goa_dir, f'{aspect}_Term')
        cur_aspect_df.to_csv(label_path, sep='\t', index=False, header=False)

        aspect_fasta_path = os.path.join(aspect_db_dir, 'AlignmentKNN.fasta')


        with open(aspect_fasta_path, 'w') as f:

            
            for entryid in unique_entryids:
                if entryid in seq_dict:
                    f.write(f">{entryid}\n{seq_dict[entryid]}\n")
                else:
                    print(f"Warning: {entryid} not found in seq_dict, skip it", file=sys.stderr)
        
        # create and diamond_db
        # print(aspect_fasta_path)
        # print(aspect_db_dir)

        # settings["diamond_path"] =  "/usr/local/bin/diamond"

        diamond_db_cmd = f'{settings["diamond_path"]} makedb --in {aspect_fasta_path} -d {os.path.join(aspect_db_dir, "AlignmentKNN")}'
        print('creating diamond_db...')
        subprocess.run(diamond_db_cmd, shell=True, check=True)       

# def goset2vec(goset:set, aspect_go2vec_dict:dict, fixed_len:bool=False):
#     if fixed_len:x
#         num_go = len(aspect_go2vec_dict)
#         vec = np.zeros(num_go)
#         for go in goset:
#             vec[aspect_go2vec_dict[go]] = 1
#         return vec
    
#     vec = list()
#     for go in goset:
#         vec.append(aspect_go2vec_dict[go])
#     vec = list(sorted(vec))
#     return vec

def goset2vec(goset: set, aspect_go2vec_dict: dict, fixed_len: bool = False):
    if fixed_len:
        num_go = len(aspect_go2vec_dict)
        vec = np.zeros(num_go)
        for go in goset:
            if go in aspect_go2vec_dict:  # Check for existence
                vec[aspect_go2vec_dict[go]] = 1
            else:
                print(f"Warning: {go} not found in aspect_go2vec_dict")

        return vec
    # print("sdfaifdpijofuoaufouewpufiuoeuwjfoejwojojeowjfjwejfjoewj")
    vec = []
    for go in goset:
        if go in aspect_go2vec_dict:  # Check for existence
            vec.append(aspect_go2vec_dict[go])
        else:
            print(f"Warning: {go} not found in aspect_go2vec_dict")
    
    return sorted(vec)


def perpare_test(test_terms_tsv:str, test_seqs_fasta:str, Data_dir:str, selected_terms_by_aspect:dict):
    """
    prepare the test data, this is mainly for the fine-tuning of precision and recall
    estimation of the model.

    Args:
        test_terms_tsv (str): path to the tsv file of the test terms
        test_seqs_fasta (str): path to the fasta file of the test sequences
        Data_dir (str): path to the directory of the data
        selected_terms_by_aspect (dict): dict of selected terms, key is aspect, value is a set of terms
    """
    test_seq_dict = get_seq_dict(test_seqs_fasta)
    present_seqs = set(test_seq_dict.keys())
    test_terms = pd.read_csv(test_terms_tsv, sep='\t')
    test_terms = prop_parents(test_terms)
    all_terms = set(selected_terms_by_aspect['BPO']) | set(selected_terms_by_aspect['CCO']) | set(selected_terms_by_aspect['MFO'])
    test_terms = test_terms[test_terms['term'].isin(all_terms)]
    test_terms = test_terms[test_terms['EntryID'].isin(present_seqs)]

    selected_entry_by_aspect = {
    'BPO': list(sorted(test_terms[test_terms['aspect'] == 'BPO']['EntryID'].unique())),
    'CCO': list(sorted(test_terms[test_terms['aspect'] == 'CCO']['EntryID'].unique())),
    'MFO': list(sorted(test_terms[test_terms['aspect'] == 'MFO']['EntryID'].unique()))
    }
    print(f'number of proteins in BPO aspect: {len(selected_entry_by_aspect["BPO"])}')
    print(f'number of proteins in CCO aspect: {len(selected_entry_by_aspect["CCO"])}')
    print(f'number of proteins in MFO aspect: {len(selected_entry_by_aspect["MFO"])}')

    # go2vec
    aspect_go2vec_dict = {
    'BPO': dict(),
    'CCO': dict(),
    'MFO': dict(),
    }
    for aspect in ['BPO', 'CCO', 'MFO']:
        for i, term in enumerate(selected_terms_by_aspect[aspect]):
            aspect_go2vec_dict[aspect][term] = i
    
    aspect_test_term_grouped_dict = {
    'BPO': test_terms[test_terms['aspect'] == 'BPO'].reset_index(drop=True).groupby('EntryID')['term'].apply(set),
    'CCO': test_terms[test_terms['aspect'] == 'CCO'].reset_index(drop=True).groupby('EntryID')['term'].apply(set),
    'MFO': test_terms[test_terms['aspect'] == 'MFO'].reset_index(drop=True).groupby('EntryID')['term'].apply(set),
    }

    out_dir = settings["TRAIN_DATA_CLEAN_DIR"]
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    for aspect in ['BPO', 'CCO', 'MFO']:
        test_dir = os.path.join(out_dir, aspect)
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
        
        aspect_test_seq_list = selected_entry_by_aspect[aspect].copy()
        aspect_test_seq_list = np.array(aspect_test_seq_list)
        aspect_test_term_matrix = np.vstack([goset2vec(aspect_test_term_grouped_dict[aspect][entry], aspect_go2vec_dict[aspect], fixed_len=True) for entry in aspect_test_seq_list])
        print(f'{aspect} label dim: {aspect_test_term_matrix.shape}')

        # convert to sparse matrix
        aspect_test_term_matrix = ssp.csr_matrix(aspect_test_term_matrix)

        np.save(os.path.join(test_dir, f'{aspect}_test_names.npy'), aspect_test_seq_list)
        ssp.save_npz(os.path.join(test_dir, f'{aspect}_test_labels.npz'), aspect_test_term_matrix)


def main(train_terms_tsv:str, train_seqs_fasta:str, Data_dir:str, \
    make_alignment_db:bool=True, 
    min_count_dict:dict=None, seed:int=42, stratifi:bool=False,
    test_terms_tsv:str=None, test_seqs_fasta:str=None):
    """
    Main function to prepare the data for training the model.

    Args:
        train_terms_tsv (str): path to the tsv file of the training terms
        train_seqs_fasta (str): path to the fasta file of the training sequences
        make_db (bool, optional): Whether to make the database. Defaults to True.
    """



    # train_seqs_fasta = "filtered_train_seq.fasta"



    # print("here55555555555")

    # print("train:", train_terms_tsv)


    

    train_seq_dict = get_seq_dict(train_seqs_fasta)



    train_terms = pd.read_csv(train_terms_tsv, sep='\t')


    # print("here33333333")
    # print(len(train_seq_dict))
    
    train_terms = prop_parents(train_terms)
    print("here444444444")
    make_alignment_db = False
    if make_alignment_db:
        make_db(Data_dir, train_terms, train_seq_dict)
        print('database created\n')





    if min_count_dict is not None:
        print("min_count_dict")
        # filter out terms with less than min_count_dictpython predict.py -w example -f example/seq.fasta --use_gpu
        bp_terms_freq = train_terms[train_terms['aspect'] == 'BPO']['term'].value_counts()
        cc_terms_freq = train_terms[train_terms['aspect'] == 'CCO']['term'].value_counts()
        mf_terms_freq = train_terms[train_terms['aspect'] == 'MFO']['term'].value_counts()
        # print("bp_terms_freq, cc_terms_freq, mf_terms_freq", bp_terms_freq, cc_terms_freq, mf_terms_freq)
        selected_bp_terms = set(bp_terms_freq[bp_terms_freq >= min_count_dict['BPO']].index)
        selected_cc_terms = set(cc_terms_freq[cc_terms_freq >= min_count_dict['CCO']].index)
        selected_mf_terms = set(mf_terms_freq[mf_terms_freq >= min_count_dict['MFO']].index)
        selected_terms = selected_bp_terms | selected_cc_terms | selected_mf_terms
        train_terms = train_terms[train_terms['term'].isin(selected_terms)]
        # print(train_terms)
        # exit()
        print(f'proteins in BPO aspect with terms frequency greater than {min_count_dict["BPO"]}: {len(selected_bp_terms)}')
        print(f'proteins in CCO aspect with terms frequency greater than {min_count_dict["CCO"]}: {len(selected_cc_terms)}')
        print(f'proteins in MFO aspect with terms frequency greater than {min_count_dict["MFO"]}: {len(selected_mf_terms)}\n')

    selected_terms_by_aspect = {
    'BPO': set(),
    'CCO': set(),
    'MFO': set(),
    }
    # print(selected_terms) we got this selected terms for training phase
    # exit()

    for term in selected_terms:
        # print(term)
        aspect = oboTools.get_aspect(term)
        # Before adding term to the set, check if it's None
        if aspect is not None:
            selected_terms_by_aspect[aspect].add(term)
        else:
            print("for term ", term)
            print(f"Warning: term is None for aspect {aspect}")

    # topological sort the terms based on parent, child relationship
    for k, v in selected_terms_by_aspect.items():
        selected_terms_by_aspect[k] = oboTools.top_sort(v)[::-1]  
    # before sort selected_terms_by_aspect is ramdon during program, after sort the order is unique

    # EntryIDs
    selected_entry_by_aspect = {
    'BPO': list(sorted(train_terms[train_terms['aspect'] == 'BPO']['EntryID'].unique())),
    'CCO': list(sorted(train_terms[train_terms['aspect'] == 'CCO']['EntryID'].unique())),
    'MFO': list(sorted(train_terms[train_terms['aspect'] == 'MFO']['EntryID'].unique()))
    }


    # go2vec
    aspect_go2vec_dict = {
    'BPO': dict(),
    'CCO': dict(),
    'MFO': dict(),
    }
    for aspect in ['BPO', 'CCO', 'MFO']:
        for i, term in enumerate(selected_terms_by_aspect[aspect]):
            aspect_go2vec_dict[aspect][term] = i

    aspect_train_term_grouped_dict = {
    'BPO': train_terms[train_terms['aspect'] == 'BPO'].reset_index(drop=True).groupby('EntryID')['term'].apply(set),
    'CCO': train_terms[train_terms['aspect'] == 'CCO'].reset_index(drop=True).groupby('EntryID')['term'].apply(set),
    'MFO': train_terms[train_terms['aspect'] == 'MFO'].reset_index(drop=True).groupby('EntryID')['term'].apply(set),
    }



    out_dir = os.path.join(Data_dir, "network_training_data")


    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if test_terms_tsv is not None and test_seqs_fasta is not None:
        perpare_test(test_terms_tsv, test_seqs_fasta, Data_dir, selected_terms_by_aspect)


    for aspect in ['BPO', 'CCO', 'MFO']:
        train_dir = os.path.join(out_dir, aspect)
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
        # write term_list
        aspect_term_list = selected_terms_by_aspect[aspect].copy()
        with open(os.path.join(train_dir, 'term_list.txt'), 'w') as f:
            for term in aspect_term_list:
                f.write(f'{term}\n')

        aspect_train_seq_list = selected_entry_by_aspect[aspect].copy()
        aspect_train_seq_list = np.array(aspect_train_seq_list)
        # print(aspect_train_seq_list)
        # exit()
        aspect_train_term_matrix = np.vstack([
            goset2vec(aspect_train_term_grouped_dict[aspect][entry], aspect_go2vec_dict[aspect], fixed_len=True) 
            for entry in aspect_train_seq_list])
        print(f'{aspect} label dim: {aspect_train_term_matrix.shape}')
        if stratifi:
            # stratified kfold
            print(f'stratified kfold for {aspect}...\nthis may take a while, for faster speed, you can use normal kfold by not setting --stratifi')

            kf = MultilabelStratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
            folds = kf.split(aspect_train_seq_list, aspect_train_term_matrix)
        else:
            kf = KFold(n_splits=5, random_state=seed, shuffle=True)
            folds = kf.split(aspect_train_seq_list)

        for i, (train_idx, val_idx) in enumerate(folds):
            print(f'creating {aspect} fold {i}...')
            train_names = aspect_train_seq_list[train_idx]

            val_names = aspect_train_seq_list[val_idx]
            aspect_fold_train_label_npy = aspect_train_term_matrix[train_idx, :]
            aspect_fold_val_label_npy = aspect_train_term_matrix[val_idx, :]




            # convert to sparse matrix
            aspect_fold_train_label_npy = ssp.csr_matrix(aspect_fold_train_label_npy)
            aspect_fold_val_label_npy = ssp.csr_matrix(aspect_fold_val_label_npy)

            ssp.save_npz(os.path.join(train_dir, f'{aspect}_train_labels_fold{i}.npz'), aspect_fold_train_label_npy)
            ssp.save_npz(os.path.join(train_dir, f'{aspect}_valid_labels_fold{i}.npz'), aspect_fold_val_label_npy)

            np.save(os.path.join(train_dir, f'{aspect}_train_names_fold{i}.npy'), train_names)
            np.save(os.path.join(train_dir, f'{aspect}_valid_names_fold{i}.npy'), val_names)

        combine_folds(aspect, train_dir)
        print(f'{aspect} done\n')            

def combine_folds(aspect, train_dir):
    # 获取所有训练和验证的文件路径
    train_names_files = sorted([os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.startswith(f"{aspect}_train_names_fold") and f.endswith(".npy")])
    train_labels_files = sorted([os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.startswith(f"{aspect}_train_labels_fold") and f.endswith(".npz")])
    valid_names_files = sorted([os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.startswith(f"{aspect}_valid_names_fold") and f.endswith(".npy")])
    valid_labels_files = sorted([os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.startswith(f"{aspect}_valid_labels_fold") and f.endswith(".npz")])

    # 合并训练数据
    train_names_combined = np.concatenate([np.load(file) for file in train_names_files])
    train_labels_combined = ssp.vstack([ssp.load_npz(file) for file in train_labels_files])

    # 合并验证数据
    valid_names_combined = np.concatenate([np.load(file) for file in valid_names_files])
    valid_labels_combined = ssp.vstack([ssp.load_npz(file) for file in valid_labels_files])

    # 保存合并后的数据
    np.save(os.path.join(train_dir, f"{aspect}_combined_train_names.npy"), train_names_combined)
    ssp.save_npz(os.path.join(train_dir, f"{aspect}_combined_train_labels.npz"), train_labels_combined)
    np.save(os.path.join(train_dir, f"{aspect}_combined_valid_names.npy"), valid_names_combined)
    ssp.save_npz(os.path.join(train_dir, f"{aspect}_combined_valid_labels.npz"), valid_labels_combined)



    print(f"Combined dataset for {aspect} saved!")

def extract_embeddings(train_seq_fasta:str):
    """extract esm embeddings from the training sequences fasta file

    Args:
        train_seq_fasta (str): path to the fasta file of the training sequences
    """
    plm = PlmEmbed(
        fasta_file=train_seq_fasta,
        working_dir=settings['tmp_dir'],
        cache_dir=settings['embedding_dir'],
        model_path=settings['esm3b_path'],
    )
    plm.extract(
        include=['mean'],
        model_type = "residue"
        # model_type = "text"
        # model_type = "esmc"
    )


    if os.path.exists(settings['tmp_dir']):
        shutil.rmtree(settings['tmp_dir'], ignore_errors=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-train","--train_terms_tsv", type=str, help="path to the tsv file of the training terms", default=None)
    parser.add_argument("-train_seqs","--train_seqs_fasta", type=str, help="path to the fasta file of the training sequences", default=None)
    parser.add_argument("-test", "--test_terms_tsv", type=str, help="path to the tsv file of the test terms", default=None)
    parser.add_argument("-test_seqs", "--test_seqs_fasta", type=str, help="path to the fasta file of the test sequences", default=None)
    parser.add_argument('-d', '--Data_dir', type=str, default=settings['DATA_DIR'], help='path to the directory of the data')
    parser.add_argument('--make_db', action='store_true', help='whether to make the alignment database')
    parser.add_argument('--min_bp', type=int, default=50, help='minimum number of BPO terms')
    parser.add_argument('--min_cc', type=int, default=20, help='minimum number of CCO terms')
    parser.add_argument('--min_mf', type=int, default=20, help='minimum number of MFO terms')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--ia', action='store_true', help='whether to calculate ia, you should have go-basic.obo in the utils/obo2csv directory, the iA.txt would store under the Data_dir')
    parser.add_argument('--stratifi', action='store_true', help='whether to use stratified multi-label in kfold')
    args = parser.parse_args()
    args.Data_dir = os.path.abspath(args.Data_dir)
    min_count_dict = {
        'BPO': args.min_bp,
        'CCO': args.min_cc,
        'MFO': args.min_mf
    }
    seed = args.seed
    make_alignment_db = args.make_db # wether to make alignment db

    if args.train_terms_tsv is not None:
        args.train_terms_tsv = os.path.abspath(args.train_terms_tsv)
    elif os.path.exists(settings['train_terms_tsv']):
        args.train_terms_tsv = settings['train_terms_tsv']
    else:
        raise Exception(f'Error: train_terms_tsv not found at {args.train_terms_tsv}, please specify the path to the training terms tsv file')
    train_terms_tsv = args.train_terms_tsv

    if args.train_seqs_fasta is not None:
        args.train_seqs_fasta = os.path.abspath(args.train_seqs_fasta)
    elif os.path.exists(settings['train_seqs_fasta']):
        args.train_seqs_fasta = settings['train_seqs_fasta']
    else:
        raise Exception(f'Error: train_seqs_fasta not found at {args.train_seqs_fasta}, please specify the path to the training sequences fasta file')
    train_seqs_fasta = args.train_seqs_fasta


    if args.test_terms_tsv is not None:
        args.test_terms_tsv = os.path.abspath(args.test_terms_tsv)
    test_terms_tsv = args.test_terms_tsv
    if args.test_seqs_fasta is not None:
        args.test_seqs_fasta = os.path.abspath(args.test_seqs_fasta)
    test_seqs_fasta = args.test_seqs_fasta

    Data_dir = args.Data_dir
    stratifi = args.stratifi

    # train_seqs_fasta = "Beprof_benchmark/BeProf_train.fasta"
    # train_terms_tsv = "Beprof_benchmark/terms_train.tsv"
    # train_seqs_fasta = "filtered_train_seq.fasta"
    # train_terms_tsv = "filtered_train_terms.tsv"


    print("start main")
    main(train_terms_tsv, train_seqs_fasta, Data_dir, make_alignment_db, min_count_dict, seed, stratifi, test_terms_tsv, test_seqs_fasta)
    print("finish main!!!!!!!!!!!!!!!!!!!!!!!")
    args.ia = False
    if args.ia:
        
        ia_file = settings['ia_file']
        ia_script = settings['ia_script']
        obo_file = settings['obo_file']
        # print(ia_file)
        # print(ia_script)
        # print(obo_file)
        # train_terms_tsv = '/Users/zjzhou/Downloads/InterLabelGO+/groundTruth.tsv'
        # exit()

        # exit(obo_file)
        obo_file = 'go-basic.obo' 
        if not os.path.exists(obo_file):
            raise Exception(f'obo file not found at {obo_file}, to calculate ia, please download the obo file from http://purl.obolibrary.org/obo/go/go-basic.obo and put it in the utils directory')
        cmd = f"python {ia_script} --annot {train_terms_tsv} --graph {obo_file} --prop -o {ia_file}"
        print('Creating IA.txt...')
        subprocess.run(cmd, shell=True, check=True)
    
    # extract embeddings
    exit("no embeddings")
    print('Extracting embeddings...')
    # train_seqs_fasta = "filtered_train_seq.fasta"
    # exit()
    # exit(train_seqs_fasta)

    extract_embeddings(train_seqs_fasta)
    if test_seqs_fasta is not None:
        extract_embeddings(test_seqs_fasta)
