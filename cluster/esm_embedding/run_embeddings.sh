 /bin/bash
#$ -l tmem=4G
#$ -l h_vmem=4G
#$ -l h_rt=2:00:00
#$ -j y
#$ -N embedjob
#$ -cwd             # or: #$ -wd /home/yourusername
#$ -V              # pass your current environment
# (Uncomment the two lines below if you want a GPU)
#$ -l gpu=true
#$ -pe gpu 1

# If there's a .source file needed, e.g. for Python 3.8, do something like:
# source /share/apps/source_files/python-3.8.5.source

python esm_residue.py /SAN/bioinf/PFP/dataset/CAFA3/CAFA3_training_data/uniprot_sprot_exp.fasta /SAN/bioinf/PFP/embeddings/esm

