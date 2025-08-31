PDBID="1mi5"
CKPT_PATH=./checkpoints/checkpoint-520k-evo-full-test1.pth
FEATURE_PATH=./dataset/${PDBID}_evoformer2_embedding.pkl
FASTA_PATH=./dataset/${PDBID}_pseudo_single.fasta
OUTDIR=./output_1mi5_evo_full_test1_0822/
mkdir -p ${OUTDIR}
python run_inference.py -c ${CKPT_PATH} -i ${FEATURE_PATH} -s ${FASTA_PATH} -o ${PDBID} --output-prefix ${OUTDIR}  -n 250 --use-gpu --use-tqdm

