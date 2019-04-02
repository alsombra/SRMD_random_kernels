#Para treino (train)

python main.py \
--scale_factor=2 \
--mode=train \
--num_workers=0 \
--total_step=50000 \
--model_save_step=1000 \
--sample_step=100 \
--num_channels=18 \
--batch_size=1 \
--image_size=64 \
--loss_function='l1' \
--trained_model=None

#Para teste no test set (test_and_error)

python main.py --scale_factor=2 --mode='test' --test_mode='pick_from_set' --num_workers=0 --total_step=300000 --model_save_step=1000 --sample_step=100 --num_channels=6 --batch_size=1 --image_size=64 --trained_model=50000.pth.tar

#Para teste fora do test set (single_test)

python main.py --scale_factor=2 --mode='test' --test_mode='single' --num_workers=0 --total_step=300000 --model_save_step=1000 --sample_step=100 --num_channels=18 --batch_size=1 --image_size=64 --trained_model=L2_50000.pth.tar --test_image_path='./test_images/0.809_lr_100.png'

#Para muitos teste fora do test set (tests) -> passe o folder no test_image_path

python main.py --scale_factor=2 --mode='test' --test_mode='many' --num_workers=0 --total_step=300000 --model_save_step=1000 --sample_step=100 --num_channels=18 --batch_size=1 --image_size=64 --trained_model=L2_50000.pth.tar --test_image_path='./results/LR_images_snapshot/'

#para evaluation (evaluate) - (does test_and_error many times)
python main.py --scale_factor=2 --mode='test' --test_mode='evaluate' --num_workers=0 --total_step=300000 --model_save_step=1000 --sample_step=100 --num_channels=18 --batch_size=1 --image_size=64 --trained_model=L2_50000.pth.tar --evaluation_step=10 --evaluation_size=2