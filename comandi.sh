
#change batch seed - cifar10

for P in 256 512 1024 2048 4096 8192

do
srun --partition gpu --qos gpu --gres gpu:1 --mem 8G --cpus-per-task 5 --time 0-4:00:00 python -m simple-models --dataset cifar10 --pte 1024 --arch cnn --L 1 --h 10 --bs 128 --dt_eff 1 --max_wall 5000 --dynamics sgd --output cifar_sb5000_t$P --ptr $P --seed_batch 5000

srun --partition gpu --qos gpu --gres gpu:1 --mem 8G --cpus-per-task 5 --time 0-4:00:00 python -m simple-models --dataset cifar10 --pte 1024 --arch cnn --L 1 --h 10 --bs 128 --dt_eff 1 --max_wall 5000 --dynamics sgd --output cifar_sb10000_t$P --ptr $P --seed_batch 10000

srun --partition gpu --qos gpu --gres gpu:1 --mem 8G --cpus-per-task 5 --time 0-4:00:00 python -m simple-models --dataset cifar10 --pte 1024 --arch cnn --L 1 --h 10 --bs 128 --dt_eff 1 --max_wall 5000 --dynamics sgd --output cifar_sb20000_t$P --ptr $P --seed_batch 20000

srun --partition gpu --qos gpu --gres gpu:1 --mem 8G --cpus-per-task 5 --time 0-4:00:00 python -m simple-models --dataset cifar10 --pte 1024 --arch cnn --L 1 --h 10 --bs 128 --dt_eff 1 --max_wall 5000 --dynamics sgd --output cifar_sb30000_t$P --ptr $P --seed_batch 30000

srun --partition gpu --qos gpu --gres gpu:1 --mem 8G --cpus-per-task 5 --time 0-4:00:00 python -m simple-models --dataset cifar10 --pte 1024 --arch cnn --L 1 --h 10 --bs 128 --dt_eff 1 --max_wall 5000 --dynamics sgd --output cifar_sb40000_t$P --ptr $P --seed_batch 40000

done

#--------------------------------------------------------------------------

#test for all the dataset - cifar10

for h in 1 2 3 4 5 6 7 8 9 10 16 20 32 40 50 64

do

srun --partition gpu --qos gpu --gres gpu:1 --mem 8G --cpus-per-task 5 --time 0-5:00:00 python -m simple-models --dataset cifar10 --pte 1024 --arch cnn --L 1 --h $h --bs 128 --dt_eff 1 --max_wall 6000 --dynamics sgd --output cifar_cnn_h$h --ptr 10000

done



#test for all the dataset - mnist

for h in 1 2 3 4 5 6 7 8 9 10 16 20 32 40 50 64

do

srun --partition gpu --qos gpu --gres gpu:1 --mem 8G --cpus-per-task 5 --time 0-5:00:00 python -m simple-models --dataset mnist --pte 1024 --arch cnn --L 1 --h $h --bs 128 --dt_eff 1 --max_wall 6000 --dynamics sgd --output mnist_cnn_h$h --ptr 10000

done




for P in 256 512 1024 2048 4096 8192

do
srun --partition gpu --qos gpu --gres gpu:1 --mem 8G --cpus-per-task 5 --time 0-4:00:00 python -m simple-models --dataset mnist --pte 1024 --arch vit --L 1 --h 128 --bs 128 --dt 0.001 --max_wall 5000 --dynamics adamw --output vit_mnist_si0_t$P --ptr $P --seed_init 0

srun --partition gpu --qos gpu --gres gpu:1 --mem 8G --cpus-per-task 5 --time 0-4:00:00 python -m simple-models --dataset mnist --pte 1024 --arch vit --L 1 --h 128 --bs 128 --dt 0.001 --max_wall 5000 --dynamics adamw --output vit_mnist_si1_t$P --ptr $P --seed_init 1

srun --partition gpu --qos gpu --gres gpu:1 --mem 8G --cpus-per-task 5 --time 0-4:00:00 python -m simple-models --dataset mnist --pte 1024 --arch vit --L 1 --h 128 --bs 128 --dt 0.001 --max_wall 5000 --dynamics adamw --output vit_mnist_si2_t$P --ptr $P --seed_init 2

srun --partition gpu --qos gpu --gres gpu:1 --mem 8G --cpus-per-task 5 --time 0-4:00:00 python -m simple-models --dataset mnist --pte 1024 --arch vit --L 1 --h 128 --bs 128 --dt 0.001 --max_wall 5000 --dynamics adamw --output vit_mnist_si3_t$P --ptr $P --seed_init 3

srun --partition gpu --qos gpu --gres gpu:1 --mem 8G --cpus-per-task 5 --time 0-4:00:00 python -m simple-models --dataset mnist --pte 1024 --arch vit --L 1 --h 128 --bs 128 --dt 0.001 --max_wall 5000 --dynamics adamw --output vit_mnist_si4_t$P --ptr $P --seed_init 4

srun --partition gpu --qos gpu --gres gpu:1 --mem 8G --cpus-per-task 5 --time 0-4:00:00 python -m simple-models --dataset cifar10 --pte 1024 --arch vit --L 1 --h 128 --bs 128 --dt 0.001 --max_wall 5000 --dynamics adamw --output vit_cifar_si0_t$P --ptr $P --seed_init 0

srun --partition gpu --qos gpu --gres gpu:1 --mem 8G --cpus-per-task 5 --time 0-4:00:00 python -m simple-models --dataset cifar10 --pte 1024 --arch vit --L 1 --h 128 --bs 128 --dt 0.001 --max_wall 5000 --dynamics adamw --output vit_cifar_si1_t$P --ptr $P --seed_init 1

srun --partition gpu --qos gpu --gres gpu:1 --mem 8G --cpus-per-task 5 --time 0-4:00:00 python -m simple-models --dataset cifar10 --pte 1024 --arch vit --L 1 --h 128 --bs 128 --dt 0.001 --max_wall 5000 --dynamics adamw --output vit_cifar_si2_t$P --ptr $P --seed_init 2

srun --partition gpu --qos gpu --gres gpu:1 --mem 8G --cpus-per-task 5 --time 0-4:00:00 python -m simple-models --dataset cifar10 --pte 1024 --arch vit --L 1 --h 128 --bs 128 --dt 0.001 --max_wall 5000 --dynamics adamw --output vit_cifar_si3_t$P --ptr $P --seed_init 3

srun --partition gpu --qos gpu --gres gpu:1 --mem 8G --cpus-per-task 5 --time 0-4:00:00 python -m simple-models --dataset cifar10 --pte 1024 --arch vit --L 1 --h 128 --bs 128 --dt 0.001 --max_wall 5000 --dynamics adamw --output vit_cifar_si4_t$P --ptr $P --seed_init 4

done









for P in 256 512 1024 2048 4096 8192

do
srun --partition gpu --qos gpu --gres gpu:1 --mem 8G --cpus-per-task 5 --time 0-4:00:00 python -m simple-models --dataset mnist --pte 1024 --arch vit --L 2 --h 128 --bs 128 --dt 0.001 --max_wall 5000 --dynamics adamw --output vit_mnist_si0_t$P --ptr $P --seed_init 0

srun --partition gpu --qos gpu --gres gpu:1 --mem 8G --cpus-per-task 5 --time 0-4:00:00 python -m simple-models --dataset mnist --pte 1024 --arch vit --L 2 --h 128 --bs 128 --dt 0.001 --max_wall 5000 --dynamics adamw --output vit_mnist_si1_t$P --ptr $P --seed_init 1

srun --partition gpu --qos gpu --gres gpu:1 --mem 8G --cpus-per-task 5 --time 0-4:00:00 python -m simple-models --dataset mnist --pte 1024 --arch vit --L 2 --h 128 --bs 128 --dt 0.001 --max_wall 5000 --dynamics adamw --output vit_mnist_si2_t$P --ptr $P --seed_init 2

srun --partition gpu --qos gpu --gres gpu:1 --mem 8G --cpus-per-task 5 --time 0-4:00:00 python -m simple-models --dataset mnist --pte 1024 --arch vit --L 2 --h 128 --bs 128 --dt 0.001 --max_wall 5000 --dynamics adamw --output vit_mnist_si3_t$P --ptr $P --seed_init 3

srun --partition gpu --qos gpu --gres gpu:1 --mem 8G --cpus-per-task 5 --time 0-4:00:00 python -m simple-models --dataset mnist --pte 1024 --arch vit --L 2 --h 128 --bs 128 --dt 0.001 --max_wall 5000 --dynamics adamw --output vit_mnist_si4_t$P --ptr $P --seed_init 4


srun --partition gpu --qos gpu --gres gpu:1 --mem 8G --cpus-per-task 5 --time 0-4:00:00 python -m simple-models --dataset mnist --pte 1024 --arch vit --L 3 --h 128 --bs 128 --dt 0.001 --max_wall 5000 --dynamics adamw --output vit_mnist_si0_t$P --ptr $P --seed_init 0

srun --partition gpu --qos gpu --gres gpu:1 --mem 8G --cpus-per-task 5 --time 0-4:00:00 python -m simple-models --dataset mnist --pte 1024 --arch vit --L 3 --h 128 --bs 128 --dt 0.001 --max_wall 5000 --dynamics adamw --output vit_mnist_si1_t$P --ptr $P --seed_init 1

srun --partition gpu --qos gpu --gres gpu:1 --mem 8G --cpus-per-task 5 --time 0-4:00:00 python -m simple-models --dataset mnist --pte 1024 --arch vit --L 3 --h 128 --bs 128 --dt 0.001 --max_wall 5000 --dynamics adamw --output vit_mnist_si2_t$P --ptr $P --seed_init 2

srun --partition gpu --qos gpu --gres gpu:1 --mem 8G --cpus-per-task 5 --time 0-4:00:00 python -m simple-models --dataset mnist --pte 1024 --arch vit --L 3 --h 128 --bs 128 --dt 0.001 --max_wall 5000 --dynamics adamw --output vit_mnist_si3_t$P --ptr $P --seed_init 3

srun --partition gpu --qos gpu --gres gpu:1 --mem 8G --cpus-per-task 5 --time 0-4:00:00 python -m simple-models --dataset mnist --pte 1024 --arch vit --L 3 --h 128 --bs 128 --dt 0.001 --max_wall 5000 --dynamics adamw --output vit_mnist_si4_t$P --ptr $P --seed_init 4

done














































