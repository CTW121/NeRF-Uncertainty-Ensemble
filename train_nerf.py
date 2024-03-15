"""
Source:
https://github.com/krrish94/nerf-pytorch
https://github.com/krrish94/nerf-pytorch/blob/master/train_nerf.py
"""

import argparse
import glob
import os
import time
from datetime import datetime

import numpy as np
import torch
import torchvision
import yaml
#from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

import matplotlib.pyplot as plt
import matplotlib.image

import imageio

from nerf import (CfgNode, get_embedding_function, get_ray_bundle, img2mse,
                  load_blender_data, load_llff_data, meshgrid_xy, models,
                  mse2psnr, run_one_iter_of_nerf)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to (.yml) config file."
    )
    parser.add_argument(
        "--load-checkpoint",
        type=str,
        default="",
        help="Path to load saved checkpoint from.",
    )
    configargs = parser.parse_args()

    # Read config file.
    cfg = None
    with open(configargs.config, "r") as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg = CfgNode(cfg_dict)
    
    # clear memory in GPU CUDA
    torch.cuda.empty_cache()

    # # (Optional:) enable this to track autograd issues when debugging
    # torch.autograd.set_detect_anomaly(True)

    # If a pre-cached dataset is available, skip the dataloader.
    USE_CACHED_DATASET = False
    train_paths, validation_paths = None, None
    images, poses, render_poses, hwf, i_split = None, None, None, None, None
    H, W, focal, i_train, i_val, i_test = None, None, None, None, None, None
    if hasattr(cfg.dataset, "cachedir") and os.path.exists(cfg.dataset.cachedir):
        train_paths = glob.glob(os.path.join(cfg.dataset.cachedir, "train", "*.data"))
        validation_paths = glob.glob(
            os.path.join(cfg.dataset.cachedir, "val", "*.data")
        )
        USE_CACHED_DATASET = True
    else:
        # Load dataset
        images, poses, render_poses, hwf = None, None, None, None
        if cfg.dataset.type.lower() == "blender":
            images, poses, render_poses, hwf, i_split = load_blender_data(
                cfg.dataset.basedir,
                half_res=cfg.dataset.half_res,
                testskip=cfg.dataset.testskip,
            )
            i_train, i_val, i_test = i_split
            
            H, W, focal = hwf

            H, W = int(H), int(W)
            hwf = [H, W, focal]
            if cfg.nerf.train.white_background:
                images = images[..., :3] * images[..., -1:] + (1.0 - images[..., -1:])
        elif cfg.dataset.type.lower() == "llff":
            images, poses, bds, render_poses, i_test = load_llff_data(
                cfg.dataset.basedir, factor=cfg.dataset.downsample_factor
            )
            hwf = poses[0, :3, -1]
            poses = poses[:, :3, :4]
            if not isinstance(i_test, list):
                i_test = [i_test]
            if cfg.dataset.llffhold > 0:
                i_test = np.arange(images.shape[0])[:: cfg.dataset.llffhold]
            i_val = i_test
            i_train = np.array(
                [
                    i
                    for i in np.arange(images.shape[0])
                    if (i not in i_test and i not in i_val)
                ]
            )
            H, W, focal = hwf
            H, W = int(H), int(W)
            hwf = [H, W, focal]
            images = torch.from_numpy(images)
            poses = torch.from_numpy(poses)

    # Seed experiment for repeatability
    seed = cfg.experiment.randomseed
    np.random.seed(seed)        # seed for random number generator of Numpy
    torch.manual_seed(seed)    # seed for random number generator of PyTorch

    # Device on which to run.
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(device)

    testimg_ = images[i_test[cfg.nerf.validation.img]].numpy()
    testimg = testimg_[:,:,:3]
    matplotlib.image.imsave('testimg_{}.png'.format(str(cfg.experiment.id)), testimg)

    encode_position_fn = get_embedding_function(
        num_encoding_functions=cfg.models.coarse.num_encoding_fn_xyz,
        include_input=cfg.models.coarse.include_input_xyz,
        log_sampling=cfg.models.coarse.log_sampling_xyz,
    )

    encode_position_fn_secondary = get_embedding_function(
        num_encoding_functions=cfg.models_secondary.coarse.num_encoding_fn_xyz,
        include_input=cfg.models_secondary.coarse.include_input_xyz,
        log_sampling=cfg.models_secondary.coarse.log_sampling_xyz,
    )

    encode_direction_fn = None
    if cfg.models.coarse.use_viewdirs:
        encode_direction_fn = get_embedding_function(
            num_encoding_functions=cfg.models.coarse.num_encoding_fn_dir,
            include_input=cfg.models.coarse.include_input_dir,
            log_sampling=cfg.models.coarse.log_sampling_dir,
        )
    
    encode_direction_fn_secondary = None
    if cfg.models_secondary.coarse.use_viewdirs:
        encode_direction_fn_secondary = get_embedding_function(
            num_encoding_functions=cfg.models_secondary.coarse.num_encoding_fn_dir,
            include_input=cfg.models_secondary.coarse.include_input_dir,
            log_sampling=cfg.models_secondary.coarse.log_sampling_dir,
        )

    # Initialize a coarse-resolution model (primary coarse model).
    model_coarse = getattr(models, cfg.models.coarse.type)(
        num_layers=cfg.models.coarse.num_layers,
        hidden_size=cfg.models.coarse.hidden_size,
        skip_connect_every=cfg.models.coarse.skip_connect_every,
        num_encoding_fn_xyz=cfg.models.coarse.num_encoding_fn_xyz,
        num_encoding_fn_dir=cfg.models.coarse.num_encoding_fn_dir,
        include_input_xyz=cfg.models.coarse.include_input_xyz,
        include_input_dir=cfg.models.coarse.include_input_dir,
        use_viewdirs=cfg.models.coarse.use_viewdirs,
    )
    model_coarse.to(device)
    print("model_coarse: \n", model_coarse)
    # coarse_state_dict = model_coarse.state_dict()
    # first_layer_weights = coarse_state_dict['layer1.weight']

    # If a fine-resolution model is specified, initialize it. (primary fine model)
    model_fine = None
    if hasattr(cfg.models, "fine"):
        model_fine = getattr(models, cfg.models.fine.type)(
            num_layers=cfg.models.fine.num_layers,
            hidden_size=cfg.models.fine.hidden_size,
            skip_connect_every=cfg.models.fine.skip_connect_every,
            num_encoding_fn_xyz=cfg.models.fine.num_encoding_fn_xyz,
            num_encoding_fn_dir=cfg.models.fine.num_encoding_fn_dir,
            include_input_xyz=cfg.models.fine.include_input_xyz,
            include_input_dir=cfg.models.fine.include_input_dir,
            use_viewdirs=cfg.models.fine.use_viewdirs,
        )
        model_fine.to(device)
    print("model_fine: \n", model_fine)

    # *** Secondary coarse models *** #
    coarse_model_secondary_list = []
    num_models = cfg.experiment.num_models_secondary
    for i in range(num_models):
        model_coarse_secondary = getattr(models, cfg.models_secondary.coarse.type)(
            num_layers=cfg.models_secondary.coarse.num_layers,
            hidden_size=cfg.models_secondary.coarse.hidden_size,
            skip_connect_every=cfg.models_secondary.coarse.skip_connect_every,
            num_encoding_fn_xyz=cfg.models_secondary.coarse.num_encoding_fn_xyz,
            num_encoding_fn_dir=cfg.models_secondary.coarse.num_encoding_fn_dir,
            include_input_xyz=cfg.models_secondary.coarse.include_input_xyz,
            include_input_dir=cfg.models_secondary.coarse.include_input_dir,
            use_viewdirs=cfg.models_secondary.coarse.use_viewdirs,
        )
        coarse_model_secondary_list.append(model_coarse_secondary)
        coarse_model_secondary_list[i].to(device)
    #     coarse_state_dict_secondary = model_coarse_secondary.state_dict()
    #     first_layer_weights = coarse_state_dict_secondary['layer1.weight']
    # for model in coarse_model_secondary_list:
    #    print(next(model.parameters()).device)

    # *** Secondary fine models *** #
    fine_model_secondary_list = []
    if hasattr(cfg.models_secondary, "fine"):
        for i in range(num_models):
            model_fine_secondary = getattr(models, cfg.models_secondary.fine.type)(
                num_layers=cfg.models_secondary.fine.num_layers,
                hidden_size=cfg.models_secondary.fine.hidden_size,
                skip_connect_every=cfg.models_secondary.fine.skip_connect_every,
                num_encoding_fn_xyz=cfg.models_secondary.fine.num_encoding_fn_xyz,
                num_encoding_fn_dir=cfg.models_secondary.fine.num_encoding_fn_dir,
                include_input_xyz=cfg.models_secondary.fine.include_input_xyz,
                include_input_dir=cfg.models_secondary.fine.include_input_dir,
                use_viewdirs=cfg.models_secondary.fine.use_viewdirs,
            )
            fine_model_secondary_list.append(model_fine_secondary)
            fine_model_secondary_list[i].to(device)
        #     fine_state_dict_secondary = model_fine_secondary.state_dict()
        #     first_layer_weights = fine_state_dict_secondary['layer1.weight']
        # for model in fine_model_secondary_list:
        #     print(next(model.parameters()).device)

    # Initialize optimizer.
    trainable_parameters = list(model_coarse.parameters())
    if model_fine is not None:
        trainable_parameters += list(model_fine.parameters())
    optimizer = getattr(torch.optim, cfg.optimizer.type)(
        trainable_parameters, lr=cfg.optimizer.lr
    )
    print("optimizer: \n", optimizer)

    optimizer_secondary_list = []
    for i in range(num_models):
        trainable_parameters_secondary = list(coarse_model_secondary_list[i].parameters())
        if fine_model_secondary_list[i] is not None:
            trainable_parameters_secondary += list(fine_model_secondary_list[i].parameters())
        optimizer_secondary = getattr(torch.optim, cfg.optimizer.type)(
            trainable_parameters_secondary, lr=cfg.optimizer.lr
        )
        optimizer_secondary_list.append(optimizer_secondary)
    # for i in range(len(optimizer_secondary_list)):
    #     print("i: ", i)
    #     print(optimizer_secondary_list[i])

    # Setup logging.
    logdir = os.path.join(cfg.experiment.logdir, cfg.experiment.id)
    os.makedirs(logdir, exist_ok=True)

    # remove output figures in the log folder
    # for file in os.listdir(logdir):
    #     os.remove(os.path.join(logdir, file))

    # writer = SummaryWriter(logdir)
    # Write out config parameters.
    with open(os.path.join(logdir, "config.yml"), "w") as f:
        f.write(cfg.dump())  # cfg, f, default_flow_style=False)

    # By default, start at iteration 0 (unless a checkpoint is specified).
    start_iter = 0

    # Load an existing checkpoint, if a path is specified.
    if os.path.exists(configargs.load_checkpoint):
        checkpoint = torch.load(configargs.load_checkpoint)
        model_coarse.load_state_dict(checkpoint["model_coarse_state_dict"])
        if checkpoint["model_fine_state_dict"]:
            model_fine.load_state_dict(checkpoint["model_fine_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        for i, coarse_model_secondary in enumerate(coarse_model_secondary_list):
            coarse_model_secondary.load_state_dict(checkpoint["model_coarse_secondary_state_dict"][i])

        for i, fine_model_secondary in enumerate(fine_model_secondary_list):
            fine_model_secondary.load_state_dict(checkpoint["model_fine_secondary_state_dict"][i])

        for i, optimizer_secondary in enumerate(optimizer_secondary_list):
            optimizer_secondary.load_state_dict(checkpoint["optimizer_secondary_state_dict"][i])

        start_iter = checkpoint["iter"]
    print("start_iter: ", start_iter)

    # TODO: Prepare raybatch tensor if batching random rays

    for i in trange(start_iter, cfg.experiment.train_iters):

        model_coarse.train()
        if model_fine:
            model_fine.train()

        rgb_coarse, rgb_fine = None, None
        target_ray_values = None
        if USE_CACHED_DATASET:
            datafile = np.random.choice(train_paths)
            cache_dict = torch.load(datafile)
            ray_bundle = cache_dict["ray_bundle"].to(device)
            ray_origins, ray_directions = (
                ray_bundle[0].reshape((-1, 3)),
                ray_bundle[1].reshape((-1, 3)),
            )
            target_ray_values = cache_dict["target"][..., :3].reshape((-1, 3))
            select_inds = np.random.choice(
                ray_origins.shape[0],
                size=(cfg.nerf.train.num_random_rays),
                replace=False,
            )
            ray_origins, ray_directions = (
                ray_origins[select_inds],
                ray_directions[select_inds],
            )
            target_ray_values = target_ray_values[select_inds].to(device)
            # ray_bundle = torch.stack([ray_origins, ray_directions], dim=0).to(device)

            # NOTE: run_one_iter_of_nerf() in nerf/train_utils.py
            rgb_coarse, _, _, rgb_fine, _, _ = run_one_iter_of_nerf(
                cache_dict["height"],
                cache_dict["width"],
                cache_dict["focal_length"],
                model_coarse,
                model_fine,
                ray_origins,
                ray_directions,
                cfg,
                mode="train",
                encode_position_fn=encode_position_fn,
                encode_direction_fn=encode_direction_fn,
            )
        else:
            img_idx = np.random.choice(i_train)
            img_target = images[img_idx].to(device)
            pose_target = poses[img_idx, :3, :4].to(device)
            ray_origins, ray_directions = get_ray_bundle(H, W, focal, pose_target)
            coords = torch.stack(
                meshgrid_xy(torch.arange(H).to(device), torch.arange(W).to(device)),
                dim=-1,
            )
            coords = coords.reshape((-1, 2))
            select_inds = np.random.choice(
                coords.shape[0], size=(cfg.nerf.train.num_random_rays), replace=False
            )
            select_inds = coords[select_inds]
            ray_origins = ray_origins[select_inds[:, 0], select_inds[:, 1], :]
            ray_directions = ray_directions[select_inds[:, 0], select_inds[:, 1], :]
            # batch_rays = torch.stack([ray_origins, ray_directions], dim=0)
            target_s = img_target[select_inds[:, 0], select_inds[:, 1], :]

            then = time.time()
            rgb_coarse, _, _, rgb_fine, _, _ = run_one_iter_of_nerf(
                H,
                W,
                focal,
                model_coarse,
                model_fine,
                ray_origins,
                ray_directions,
                cfg,
                mode="train",
                encode_position_fn=encode_position_fn,
                encode_direction_fn=encode_direction_fn,
            )
            target_ray_values = target_s

            """
            Secondary coarse and fine models
            """

            for k in range(cfg.experiment.num_models_secondary):
                coarse_model_secondary_list[k].train()
                fine_model_secondary_list[k].train()

                rgb_coarse_secondary, _, _, rgb_fine_secondary, _, _ = run_one_iter_of_nerf(
                    H,
                    W,
                    focal,
                    coarse_model_secondary_list[k],
                    fine_model_secondary_list[k],
                    ray_origins,
                    ray_directions,
                    cfg,
                    mode="train",
                    encode_position_fn=encode_position_fn_secondary,
                    encode_direction_fn=encode_direction_fn_secondary,
                )

                # https://discuss.pytorch.org/t/only-update-weight-of-one-model-in-training/140223

                coarse_loss_secondary = torch.nn.functional.mse_loss(
                    rgb_coarse_secondary[..., :3], target_ray_values[..., :3]
                )

                fine_loss_secondary = None
                if rgb_fine_secondary is not None:
                    fine_loss_secondary = torch.nn.functional.mse_loss(
                        rgb_fine_secondary[..., :3], target_ray_values[..., :3]
                    )
                
                loss_secondary = 0.0
                loss_secondary = coarse_loss_secondary + (fine_loss_secondary if fine_loss_secondary is not None else 0.0)
                loss_secondary.backward()
                optimizer_secondary_list[k].step()
                optimizer_secondary_list[k].zero_grad()



        coarse_loss = torch.nn.functional.mse_loss(
            rgb_coarse[..., :3], target_ray_values[..., :3]
        )
        fine_loss = None
        if rgb_fine is not None:
            fine_loss = torch.nn.functional.mse_loss(
                rgb_fine[..., :3], target_ray_values[..., :3]
            )
        # loss = torch.nn.functional.mse_loss(rgb_pred[..., :3], target_s[..., :3])
        loss = 0.0
        # if fine_loss is not None:
        #     loss = fine_loss
        # else:
        #     loss = coarse_loss
        loss = coarse_loss + (fine_loss if fine_loss is not None else 0.0)
        loss.backward()
        #psnr = mse2psnr(loss.item())
        optimizer.step()
        optimizer.zero_grad()


        # Learning rate updates
        num_decay_steps = cfg.scheduler.lr_decay * 1000
        lr_new = cfg.optimizer.lr * (
            cfg.scheduler.lr_decay_factor ** (i / num_decay_steps)
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_new
        
        for opt in optimizer_secondary_list:
            for param_group in opt.param_groups:
                param_group["lr"] = lr_new

        # if i % cfg.experiment.print_every == 0 or i == cfg.experiment.train_iters - 1:
        #     tqdm.write(
        #         "[TRAIN] Iter: "
        #         + str(i)
        #         + " Loss: "
        #         + str(loss.item())
        #         + " PSNR: "
        #         + str(psnr)
        #     )
        # writer.add_scalar("train/loss", loss.item(), i)
        # writer.add_scalar("train/coarse_loss", coarse_loss.item(), i)
        # if rgb_fine is not None:
        #     writer.add_scalar("train/fine_loss", fine_loss.item(), i)
        # writer.add_scalar("train/psnr", psnr, i)

        # Validation
        if (
            i % cfg.experiment.validate_every == 0
            or i == cfg.experiment.train_iters - 1
        ):
            # tqdm.write("[VAL] =======> Iter: " + str(i))
            model_coarse.eval()
            if model_fine:
                model_fine.eval()

            start = time.time()
            with torch.no_grad():
                rgb_coarse, rgb_fine = None, None
                target_ray_values = None
                if USE_CACHED_DATASET:
                    datafile = np.random.choice(validation_paths)
                    cache_dict = torch.load(datafile)
                    rgb_coarse, _, _, rgb_fine, _, _ = run_one_iter_of_nerf(
                        cache_dict["height"],
                        cache_dict["width"],
                        cache_dict["focal_length"],
                        model_coarse,
                        model_fine,
                        cache_dict["ray_origins"].to(device),
                        cache_dict["ray_directions"].to(device),
                        cfg,
                        mode="validation",
                        encode_position_fn=encode_position_fn,
                        encode_direction_fn=encode_direction_fn,
                    )
                    target_ray_values = cache_dict["target"].to(device)
                else:
                    img_idx = np.random.choice(i_val)
                    img_target = images[img_idx].to(device)
                    pose_target = poses[img_idx, :3, :4].to(device)
                    ray_origins, ray_directions = get_ray_bundle(
                        H, W, focal, pose_target
                    )
                    rgb_coarse, _, _, rgb_fine, _, _ = run_one_iter_of_nerf(
                        H,
                        W,
                        focal,
                        model_coarse,
                        model_fine,
                        ray_origins,
                        ray_directions,
                        cfg,
                        mode="validation",
                        encode_position_fn=encode_position_fn,
                        encode_direction_fn=encode_direction_fn,
                    )
                    target_ray_values = img_target
                coarse_loss = img2mse(rgb_coarse[..., :3], target_ray_values[..., :3])
                loss, fine_loss = 0.0, 0.0
                if rgb_fine is not None:
                    fine_loss = img2mse(rgb_fine[..., :3], target_ray_values[..., :3])
                    loss = fine_loss
                else:
                    loss = coarse_loss
                loss = coarse_loss + fine_loss
                psnr = mse2psnr(loss.item())
                # writer.add_scalar("validation/loss", loss.item(), i)
                # writer.add_scalar("validation/coarse_loss", coarse_loss.item(), i)
                # writer.add_scalar("validataion/psnr", psnr, i)
                # writer.add_image(
                #     "validation/rgb_coarse", cast_to_image(rgb_coarse[..., :3]), i
                # )
                # if rgb_fine is not None:
                #     writer.add_image(
                #         "validation/rgb_fine", cast_to_image(rgb_fine[..., :3]), i
                #     )
                #     writer.add_scalar("validation/fine_loss", fine_loss.item(), i)
                # writer.add_image(
                #     "validation/img_target",
                #     cast_to_image(target_ray_values[..., :3]),
                #     i,
                # )
                # tqdm.write(
                #     "Validation loss: "
                #     + str(loss.item())
                #     + " Validation PSNR: "
                #     + str(psnr)
                #     + " Time: "
                #     + str(time.time() - start)
                # )
                tqdm.write("[validation] Loss: {:.4f} | PSNR: {:.4f}".format(loss.item(), psnr))

        if i % cfg.experiment.save_every == 0 or i == cfg.experiment.train_iters - 1:
            checkpoint_dict = {
                "iter": i,
                "model_coarse_state_dict": model_coarse.state_dict(),
                "model_fine_state_dict": None
                if not model_fine
                else model_fine.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
                "psnr": psnr,
                "model_coarse_secondary_state_dict": [],
                "model_fine_secondary_state_dict": [],
                "optimizer_secondary_state_dict": [],
            }

            for coarse_model_secondary in coarse_model_secondary_list:
                checkpoint_dict["model_coarse_secondary_state_dict"].append(coarse_model_secondary.state_dict())

            for fine_model_secondary in fine_model_secondary_list:
                checkpoint_dict["model_fine_secondary_state_dict"].append(fine_model_secondary.state_dict())

            for optimizer_secondary in optimizer_secondary_list:
                checkpoint_dict["optimizer_secondary_state_dict"].append(optimizer_secondary.state_dict())

            torch.save(
                checkpoint_dict,
                os.path.join(logdir, "checkpoint" + str(i).zfill(5) + ".ckpt"),
            )
            #tqdm.write("================== Saved Checkpoint =================")
        
        # Test
        if i % cfg.experiment.print_every == 0 or i == cfg.experiment.train_iters - 1:
            model_coarse.eval()
            if model_fine:
                model_fine.eval()

            coarse_model_secondary_list[0].eval()
            if model_fine:
                fine_model_secondary_list[0].eval()

            with torch.no_grad():
                rgb_coarse, rgb_fine = None, None
                img_idx = i_test[cfg.nerf.validation.img]
                img_test = images[img_idx].to(device)
                pose_test = poses[img_idx, :3, :4].to(device)
                ray_origins, ray_directions = get_ray_bundle(
                    H, W, focal, pose_test
                )
                # print("model_fine: \n", model_fine)
                # if model_fine == None:
                #     print("model fine is None")
                # else:
                #     print("model fine is NOT None")
                rgb_coarse, _, _, rgb_fine, _, _ = run_one_iter_of_nerf(
                    H,
                    W,
                    focal,
                    model_coarse,
                    model_fine,
                    ray_origins,
                    ray_directions,
                    cfg,
                    mode="validation",
                    encode_position_fn=encode_position_fn,
                    encode_direction_fn=encode_direction_fn,
                )
                
                savefile = os.path.join(logdir, f"fine_{i:06d}.png")
                imageio.imwrite(
                    savefile, cast_to_image(rgb_fine[..., :3])
                )

                rgb_coarse_secondary, _, _, rgb_fine_secondary, _, _ = run_one_iter_of_nerf(
                    H,
                    W,
                    focal,
                    coarse_model_secondary_list[0],
                    fine_model_secondary_list[0],
                    ray_origins,
                    ray_directions,
                    cfg,
                    mode="validation",
                    encode_position_fn=encode_position_fn_secondary,
                    encode_direction_fn=encode_direction_fn_secondary,
                )

                savefile_secondary = os.path.join(logdir, f"fine_{i:06d}_secondary.png")
                imageio.imwrite(
                    savefile_secondary, cast_to_image(rgb_fine_secondary[..., :3])
                )

    print("Done!")


def cast_to_image(tensor):
    # Input tensor is (H, W, 3). Convert to (3, H, W).
    tensor = tensor.permute(2, 0, 1)
    # Conver to PIL Image and then np.array (output shape: (H, W, 3))
    img = np.array(torchvision.transforms.ToPILImage()(tensor.detach().cpu()))
    # Map back to shape (3, H, W), as tensorboard needs channels first.
    #img = np.moveaxis(img, [-1], [0])
    return img


if __name__ == "__main__":
    start_time = datetime.now()

    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print("Start time: ", dt_string)
    
    main()

    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print("End time: ", dt_string)
    
    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))