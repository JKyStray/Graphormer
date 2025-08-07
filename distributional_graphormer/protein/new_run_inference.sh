PDBID="A6"
CKPT_PATH=./checkpoints/checkpoint-520k-adaptor-test5-centered.pth
INIT_STATE_PATH=./dataset/${PDBID}_init_state.npz
FEATURE_PATH=./dataset/${PDBID}_tfold_v1.pkl
FASTA_PATH=./dataset/${PDBID}_pseudo_single.fasta
OUTDIR=./output_A6_adaptor_test5_centered_0805_2/
mkdir -p ${OUTDIR}
python run_inference.py -c ${CKPT_PATH} -i ${FEATURE_PATH} -s ${FASTA_PATH} -o ${PDBID} --output-prefix ${OUTDIR} -n 5 --use-gpu --use-tqdm
