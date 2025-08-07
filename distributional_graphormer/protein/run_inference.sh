PDBID="A6"
CKPT_PATH=./checkpoints/checkpoint-520k-evo-test2.pth
FEATURE_PATH=./dataset/${PDBID}_evoformer2_embedding.pkl
FASTA_PATH=./dataset/${PDBID}_pseudo_single.fasta
OUTDIR=./output_evo2_trained_test2_0803/
mkdir -p ${OUTDIR}
python run_inference.py -c ${CKPT_PATH} -i ${FEATURE_PATH} -s ${FASTA_PATH} -o A6 --output-prefix ${OUTDIR}  -n 10 --use-gpu --use-tqdm

