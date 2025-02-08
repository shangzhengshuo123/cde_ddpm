import time

from Diffusion.Train import train, eval

def main(model_config=None):
    modelConfig = {
        "state": "eval",  # train or eval
        "epoch": 4000,
        "batch_size": 4,
        "T": 1000,
        "channel": 128,
        "channel_mult": [1, 2, 3, 4],
        "attn": [2],
        "num_res_blocks": 2,
        "dropout": 0.15,
        "lr": 0.0001,
        "multiplier": 2.,
        "beta_1": 1e-4,
        "beta_T": 0.02,
        "img_size": 128,
        "grad_clip": 1.,
        "device": "cuda:0",
        "training_load_weight": None,
        "save_weight_dir": "./Checkpoints/1/",
        "test_load_weight": "ckpt_NA3999_.pt",
        "sampled_dir": "./SampleData/1/",
        "sampledNoisyImgName": "NoisyNoGuidenceImgs.png",
        "sampledImgName": "111NoGuidenceImgs.png",
        "nrow": 1
    }
    if model_config is not None:
        modelConfig = model_config
    if modelConfig["state"] == "train":
        train(modelConfig)
    else:
        eval(modelConfig)


if __name__ == '__main__':
    # main()
    start_time = time.time()
    main()
    end_time = time.time()
    run_time = end_time - start_time
    print("运行时间：" + str(int(run_time)) + "秒")