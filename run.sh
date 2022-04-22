cd ~/CSCI2240-Final
rm -rf ./logs/blender_paper_lego/*
python3 run_nerf.py --config configs/lego.txt --i_testset 1000 --render_factor 2 --chunk 16384 --netchunk 16384
