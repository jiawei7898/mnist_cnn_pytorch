import os
import subprocess
import sys
import logging
from pathlib import Path

# 配置日志记录
logging.basicConfig(
    filename='pipeline.log',
    filemode='w',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)
logger = logging.getLogger(__name__)


def run_command(command, cwd=None):
    """运行系统命令并捕获输出"""
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            cwd=cwd  # 指定工作目录
        )
        stdout, stderr = process.communicate()
        return_code = process.returncode

        if return_code == 0:
            logger.info(f"Command '{command}' executed successfully.")
            if stdout:
                logger.debug(stdout.decode())
        else:
            logger.error(f"Command '{command}' failed with return code {return_code}.")
            if stderr:
                logger.error(stderr.decode())
            raise subprocess.CalledProcessError(return_code, command)

    except Exception as e:
        logger.error(f"An error occurred while running the command: {e}")
        raise


def check_file_exists(file_path):
    """检查文件是否存在"""
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")


def main():
    # 获取当前脚本所在的目录
    current_dir = Path(__file__).resolve().parent

    # 定义训练和测试脚本的路径
    train_script = current_dir / 'train.py'
    test_script = current_dir / 'test.py'

    # 检查训练和测试脚本是否存在
    check_file_exists(train_script)
    check_file_exists(test_script)

    print("Starting the training process...")
    logger.info("Starting the training process...")
    try:
        run_command(f"python {train_script}", cwd=current_dir)
        logger.info("Training completed successfully.")
        print("Training completed. Starting the testing process...")
        logger.info("Starting the testing process...")
        run_command(f"python {test_script}", cwd=current_dir)
        logger.info("Testing completed successfully.")
        print("All processes completed.")
        logger.info("All processes completed.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Process failed: {e}")
        print(f"Process failed: {e}", file=sys.stderr)
        sys.exit(1)  # 退出程序，返回非零状态码表示失败
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
