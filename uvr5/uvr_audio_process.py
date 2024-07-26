import os
import ffmpeg
import torch
from uvr5.mdxnet import MDXNetDereverb
from uvr5.vr import AudioPre, AudioPreDeEcho
from loguru import logger


class uvr:
    def __init__(
        self,
        model_weights_root,
        temp_dir,
        device="cuda",
        is_half=False,
    ):
        self.model_weights_root = model_weights_root
        self.device = device
        self.is_half = is_half
        self.temp_dir = temp_dir

    # 定义UVR处理函数
    def process(
        self,
        model_name,
        input_dir,
        vocal_output_dir,
        instrumental_output_dir,
        denoise_strength=10,
        output_format="wav",
    ):
        """
        model_name: 模型名
        input_dir: 输入文件夹路径
        vocal_output_dir: 输出人声文件夹路径
        instrumental_output_dir: 输出非人声文件夹路径
        denoise_strength: 降噪强度
        output_format: 输出音频格式
        """
        os.makedirs(vocal_output_dir, exist_ok=True)
        os.makedirs(instrumental_output_dir, exist_ok=True)

        try:
            is_hp3 = "HP3" in model_name
            if model_name == "onnx_dereverb_By_FoxJoy":
                processor = MDXNetDereverb(15)
            else:
                processor_class = (
                    AudioPre if "DeEcho" not in model_name else AudioPreDeEcho
                )
                processor = processor_class(
                    agg=int(denoise_strength),
                    model_path=os.path.join(
                        self.model_weights_root, model_name + ".pth"
                    ),
                    device=self.device,
                    is_half=self.is_half,
                )

            audio_files = [
                os.path.join(input_dir, name) for name in os.listdir(input_dir)
            ]

            logger.info(f"""
使用模型: {model_name}
输入文件夹: {input_dir}
输入文件数量: {len(audio_files)}
输出人声文件夹: {vocal_output_dir}
输出非人声文件夹: {instrumental_output_dir}
降噪强度: {denoise_strength}
输出音频格式: {output_format}
""")

            for file_path in audio_files:
                if not os.path.isfile(file_path):
                    continue
                need_reformat = 1
                processed = 0
                try:
                    info = ffmpeg.probe(file_path, cmd="ffprobe")
                    if (
                        info["streams"][0]["channels"] == 2
                        and info["streams"][0]["sample_rate"] == "44100"
                    ):
                        need_reformat = 0
                        processor._path_audio_(
                            file_path,
                            instrumental_output_dir,
                            vocal_output_dir,
                            output_format,
                            is_hp3,
                        )
                        processed = 1
                except Exception as e:
                    need_reformat = 1
                    logger.error(f"音频信息获取失败，需重新格式化: {e}", exc_info=True)
                if need_reformat == 1:
                    tmp_path = "%s/%s.reformatted.wav" % (
                        self.temp_dir,
                        os.path.basename(file_path),
                    )
                    os.system(
                        f'ffmpeg -i "{file_path}" -vn -acodec pcm_s16le -ac 2 -ar 44100 "{tmp_path}" -y'
                    )
                    file_path = tmp_path
                try:
                    if processed == 0:
                        processor._path_audio_(
                            file_path,
                            instrumental_output_dir,
                            vocal_output_dir,
                            output_format,
                            is_hp3,
                        )

                except Exception as e:
                    logger.error(f"处理音频时出错: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"处理过程中发生错误: {e}", exc_info=True)
        finally:
            try:
                if model_name == "onnx_dereverb_By_FoxJoy":
                    del processor.pred.model
                    del processor.pred.model_
                else:
                    del processor.model
                    del processor
            except Exception as e:
                logger.error(f"清理模型缓存时出错: {e}", exc_info=True)
            logger.info("清理GPU缓存")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
