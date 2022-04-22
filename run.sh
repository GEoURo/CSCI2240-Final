cd ~/CSCI2240-Final
rm -rf ./logs/blender_paper_lego/*
python3 run_nerf.py --config configs/lego.txt --i_testset 5000 --i_video 5000 --chunk 16384 --netchunk 16384
