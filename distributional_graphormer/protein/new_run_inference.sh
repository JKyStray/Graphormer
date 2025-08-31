PDBID="1mi5"
CKPT_PATH=./checkpoints/checkpoint-520k-direct-new-further.pth
INIT_STATE_PATH=./dataset/${PDBID}_init_state.npz
FEATURE_PATH=./dataset/${PDBID}_tfold_v1.pkl
FASTA_PATH=./dataset/${PDBID}_pseudo_single.fasta
OUTDIR=./output_1mi5_direct_new_further_0823/
mkdir -p ${OUTDIR}
python run_inference.py -c ${CKPT_PATH} -i ${FEATURE_PATH} -s ${FASTA_PATH} -o ${PDBID} --output-prefix ${OUTDIR} -n 250 --use-gpu --use-tqdm
