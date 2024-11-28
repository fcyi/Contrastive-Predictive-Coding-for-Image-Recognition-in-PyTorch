import torch
import datetime
import os

from torch.utils.data import DataLoader
from imagenet_dataset import get_imagenet_datasets
from helper_functions import dot, dot_norm, dot_norm_exp, norm_euclidian, get_random_patches, get_patch_tensor_from_image_batch
from helper_functions import write_csv_stats


def run_context_predictor(args, res_encoder_model, context_predictor_model, models_store_path):

    print("RUNNING CONTEXT PREDICTOR TRAINING")

    stats_csv_path = os.path.join(models_store_path, "pred_stats.csv")

    dataset_train, dataset_test = get_imagenet_datasets(args.image_folder, num_classes = args.num_classes)

    def get_random_patch_loader():
        return DataLoader(dataset_train, args.num_random_patches, shuffle=True)

    random_patch_loader = get_random_patch_loader()
    data_loader_train = DataLoader(dataset_train, args.sub_batch_size, shuffle = True)

    params = list(res_encoder_model.parameters()) + list(context_predictor_model.parameters())
    optimizer = torch.optim.Adam(params = params, lr=0.00001)

    sub_batches_processed = 0
    batch_loss = 0
    sum_batch_loss = 0 
    best_batch_loss = 1e10

    z_vect_similarity = dict()

    loaderIdx_ = 0
    loaderTotal_ = len(data_loader_train)
    for batch in data_loader_train:
        loaderIdx_ += 1
        # plt.imshow(img_arr.permute(1,2,0))
        # fig, axes = plt.subplots(7,7)

        img_batch = batch['image']
        patch_batch = get_patch_tensor_from_image_batch(img_batch).to(args.device)
        batch_size = len(img_batch)

        # 利用编码网络对图像上截取到的图像块同一进行编码
        patches_encoded = res_encoder_model.forward(patch_batch)
        # 对编码后的图像块沿着其在图像中的位置进行维度调整， B*49,C -> B,7,7,C
        patches_encoded = patches_encoded.view(batch_size, 7,7,-1)
        # 对编码后的图像块进一步维度调整，B,7,7,C -> B,C,7,7
        patches_encoded = patches_encoded.permute(0,3,1,2)

        for i in range(2):
            patches_return = get_random_patches(random_patch_loader, args.num_random_patches)
            if patches_return['is_data_loader_finished']:
                random_patch_loader = get_random_patch_loader()
            else:
                random_patches = patches_return['patches_tensor'].to(args.device)

        # enc_random_patches = resnet_encoder.forward(random_patches).detach()
        enc_random_patches = res_encoder_model.forward(random_patches)  # 对从另一个数据迭代器随机抽取的图像块进行编码

        # TODO: vectorize the context_predictor_model - stack all 3x3 contexts together
        predictions, locations = context_predictor_model.forward(patches_encoded)
        losses = []

        # 对每个编码块上预测的6个位置的编码（后三行以及后三列）进行遍历
        for b in range(len(predictions)//batch_size):
            # 逐批抽取预测编码
            b_idx_start = b*batch_size
            b_idx_end = (b+1)*batch_size

            # 抽取预测编码在编码块上的位置索引
            p_y = locations[b_idx_start][0]
            p_x = locations[b_idx_start][1]

            target = patches_encoded[:, :, p_y, p_x]  # 卷积上下文网络预测的编码所对应的真实信息（即相应图像块通过编码网络获取到的编码）
            pred = predictions[b_idx_start:b_idx_end]  # 卷积上下文网络预测的编码

            dot_norm_val = dot_norm_exp(pred.detach().to('cpu'), target.detach().to('cpu'))
            euc_loss_val = norm_euclidian(pred.detach().to('cpu'), target.detach().to('cpu'))

            good_term_dot = dot(pred, target)  # 计算预测编码与对应的真实编码之间的内积（即预测编码与正样本之间的内积）
            dot_terms = [torch.unsqueeze(good_term_dot,dim=0)]

            for random_patch_idx in range(args.num_random_patches):
                # 计算来每一个自于其他图像的随机图像块的编码与批次中每一个预测编码之间的内积（即预测编码与负样本之间的内积）
                bad_term_dot = dot(pred, enc_random_patches[random_patch_idx:random_patch_idx+1])
                dot_terms.append(torch.unsqueeze(bad_term_dot, dim=0))

            # 对每个预测编码与每个正样本或负样本之间的内积沿着正负样本索引方向进行拼接（先正样本，再负样本），
            # 在沿着正负样本索引方向求对数概率。
            # log_softmax[i,j]表示第j个预测编码与第i个样本编码之间的内积的对数概率值
            log_softmax = torch.log_softmax(torch.cat(dot_terms, dim=0), dim=0)
            # 将log_softmax的第一行数据的负数作为损失，损失的最小化其实就是在最大化预测编码与对应正样本编码的内积，
            # 由于采用基于内积定义的对数概率，因此当预测编码与正样本编码的内积增大，则其与负样本编码的内积就会减小。
            losses.append(-log_softmax[0,])
            # losses.append(-torch.log(good_term/divisor))

        # 可以发现，虽然进行梯度回传，但是并没有进行参数更新，这是为了累积到batch_size个损失后，再用计算图上累积的梯度进行参数更新
        # 此外，此处的损失是将编码网络和全局上下文网络放在一起进行更新
        loss = torch.mean(torch.cat(losses))  # 首先将losses中的各项拼接成一个一维张量，在对该张量求均值，从而得到平均方差
        loss.backward()

        # loss = torch.sum(torch.cat(losses))
        # loss.backward()

        # 个人感觉此处的batch_loss相当于是sum_batch_loss // batch_size，这其实就是用于更新参数的损失。
        # 因为计算图上有损失累积、梯度累积的过程
        sub_batches_processed += img_batch.shape[0]
        batch_loss += loss.detach().to('cpu')
        sum_batch_loss += torch.sum(torch.cat(losses).detach().to('cpu'))

        if sub_batches_processed >= args.batch_size:
            print("{}\t/{}".format(loaderIdx_, loaderTotal_))
            optimizer.step()
            optimizer.zero_grad()

            print(f"{datetime.datetime.now()} Loss: {batch_loss}")
            print(f"{datetime.datetime.now()} SUM Loss: {sum_batch_loss}")

            torch.save(res_encoder_model.state_dict(), os.path.join(models_store_path, "last_res_ecoder_weights.pt"))
            torch.save(context_predictor_model.state_dict(), os.path.join(models_store_path, "last_context_predictor_weights.pt"))

            if best_batch_loss > batch_loss:
                best_batch_loss = batch_loss
                torch.save(res_encoder_model.state_dict(), os.path.join(models_store_path, "best_res_ecoder_weights.pt"))
                torch.save(context_predictor_model.state_dict(), os.path.join(models_store_path, "best_context_predictor_weights.pt"))

            for key, cos_similarity_tensor in z_vect_similarity.items():
                print(f"Mean cos_sim for class {key} is {cos_similarity_tensor.mean()} . Number: {cos_similarity_tensor.size()}")

            z_vect_similarity = dict()

            stats = dict(
                batch_loss = batch_loss,
                sum_batch_loss = sum_batch_loss
            )
            write_csv_stats(stats_csv_path, stats)

            sub_batches_processed = 0
            batch_loss = 0
            sum_batch_loss = 0

