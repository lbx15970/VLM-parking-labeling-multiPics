import os
from runners.seed20_runner import Seed20Runner

def test_seed2():
    for env_file in [".env", "env"]:
        if os.path.exists(env_file):
            with open(env_file) as f:
                for line in f:
                    if "=" in line and not line.startswith(("#", ";")):
                        k, v = line.strip().split("=", 1)
                        if "#" in v:
                            v = v.split("#")[0]
                        os.environ[k.strip()] = v.strip().strip("\"'")

    api_key = os.getenv("SEED20_API_KEY") or os.getenv("ARK_API_KEY")
    ep = os.getenv("SEED20_EP", "ep-20260304172343-7thnh")
    model_id = os.getenv("SEED20_MODEL_ID", "doubao-seed-2-0-pro-preview-260215")

    runner = Seed20Runner(api_key=api_key, ep=ep, model_id=model_id, base_url="https://ark.cn-beijing.volces.com/api/v3")
    
    # 找一张图测试
    img_path = "data/images/00001.jpg"
    print("Testing Seed 2.0 Pro...")
    try:
        res = runner.run("你好，请描述这张图片", img_path, 120)
        print("Success!")
        print(res.choices[0].message.content)
    except Exception as e:
        print("Failed:", e)

if __name__ == "__main__":
    test_seed2()
