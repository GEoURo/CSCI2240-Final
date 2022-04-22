cd ~/CSCI2240-Final
rm -rf ./logs/blender_paper_lego/*
python3 run_nerf.py --config configs/lego.txt --chunk 16384 --netchunk 16384
