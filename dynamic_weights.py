from flask import Flask, request, jsonify
import clip
import torch
from torch import nn
import json
import time
from typing import List, Dict
from flask.json.provider import DefaultJSONProvider
import hashlib
from user_agents import parse
import redis
import geoip2
from flask import request
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import base64
import cv2
from PIL import Image
import GPUtil
import threading

# 自定义 JSON 编码器，禁用 ASCII 转义
class CustomJSONProvider(DefaultJSONProvider):
    ensure_ascii = False

app = Flask(__name__)
# 使用自定义的 JSON 编码器
app.json = CustomJSONProvider(app)

# 获取设备类型
def get_device_type():
    user_agent = parse(request.headers.get("User-Agent", ""))
    return "mobile" if user_agent.is_mobile else "PC"

# 获取地理位置
def get_geo_location(ip):
    try:
        reader = geoip2.database.Reader('/path/to/GeoLite2-City.mmdb')
        response = reader.city(ip)
        return response.country.iso_code  # 返回国家代码，如"CN"
    except Exception as e:
        return "unknown"

# 上下文特征合并
def encode_context():
    device = get_device_type()
    geo = get_geo_location(request.remote_addr)
    # 示例编码逻辑
    device_encoder = OneHotEncoder(handle_unknown='ignore')
    geo_encoder = OneHotEncoder(handle_unknown='ignore')

    # 训练编码器（需根据实际类别初始化）
    device_encoder.fit(np.array([["mobile"], ["PC"]]))
    geo_encoder.fit(np.array([["CN"], ["US"], ["unknown"]]))

    # 上下文特征合并
    device_vec = device_encoder.transform([[device]]).toarray()[0]
    geo_vec = geo_encoder.transform([[geo]]).toarray()[0]
    return np.concatenate([device_vec, geo_vec])

# 加载CLIP模型
device = "cuda" if torch.cuda.is_available() else "cpu"
try:
    clip_model, preprocess = clip.load('ViT-B/32', device=device)
except Exception as e:
    print(f"模型加载失败: {e}")
    clip_model = None

# 将Base64编码的图像转换为OpenCV格式
def base64_to_opencv(base64_string):
    img_data = base64.b64decode(base64_string)
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

# 根据Base64图像获取CLIP编码
def extract_image_features(base64_image):
    try:
        # 将Base64图像转换为OpenCV格式
        cv_image = base64_to_opencv(base64_image)
        # 将OpenCV图像转换为PIL图像
        pil_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        # 对图像进行预处理
        image_input = preprocess(pil_image).unsqueeze(0).to(device)
        # 使用CLIP模型进行编码
        with torch.no_grad():
            image_features = clip_model.encode_image(image_input)
        return image_features
    except Exception as e:
        print(f"图像特征提取失败: {e}")
        return torch.randn(1, 512).to(device)



def calculate_semantic_score(value1: Dict, value2: Dict) -> float:
    """
    计算两个值的语义对齐得分，支持文本、图像、向量类型
    :param value1: 包含类型和值的字典
    :param value2: 包含类型和值的字典
    :return: 语义对齐得分
    """

    if value1['type'] == 'text':
        code1 = clip.tokenize(value1['value']).to(device)
        with torch.no_grad():
            code1 = clip_model.encode_text(code1)
    elif value1['type'] == 'image':
        code1 =  extract_image_features(value1['value']['base64'])
    elif value1['type'] == 'vector':
        code1 = torch.tensor(value1['value']).to(device)
    else:
        return 0.1
    
    if value2['type'] == 'text':
        code2 = clip.tokenize(value2['value']).to(device)
        with torch.no_grad():
            code2 = clip_model.encode_text(code2)
    elif value2['type'] == 'image':
        code2 =  extract_image_features(value2['value']['base64'])
    elif value2['type'] == 'vector':
        code2 = torch.tensor(value2['value']).to(device)
    else:
        return 0.1

  # 检查维度并调整
    if code1.dim() == 1:
        code1 = code1.unsqueeze(0)
    if code2.dim() == 1:
        code2 = code2.unsqueeze(0)

    # 获取两个张量的特征维度
    dim1 = code1.size(1)
    dim2 = code2.size(1)

    # 确定较大的维度
    max_dim = max(dim1, dim2)

    # 填充较小的张量
    if dim1 < max_dim:
        padding = torch.zeros((code1.size(0), max_dim - dim1), device=device)
        code1 = torch.cat([code1, padding], dim=1)
    elif dim2 < max_dim:
        padding = torch.zeros((code2.size(0), max_dim - dim2), device=device)
        code2 = torch.cat([code2, padding], dim=1)

    # 确保输入的张量是浮点型
    code1 = code1.to(torch.float32)
    code2 = code2.to(torch.float32)

    similarity = nn.functional.cosine_similarity(code1, code2).item()
    return similarity



def adjust_weights(attributes: List[Dict], temperature: float = 1.2) -> tuple[Dict[str, float], List[str]]:
    """
    根据语义得分和温度参数调整权重
    :param attributes: 属性列表
    :param temperature: 温度参数
    :return: 调整后的权重字典和调整日志
    """
    alpha = 2.0
    final_weights = {}
    adjustment_log = []
    # 记录初始权重
    initial_weight_str = "初始权重: " + "→".join([f"{attr['value']}({attr['initial_weight']})" for attr in attributes])
    adjustment_log.append(initial_weight_str)

    # 计算分母 Σ(Si^α)
    denominator = 0
    for attr in attributes:
        semantic_score = calculate_semantic_score(attr, attributes[0])
        denominator += semantic_score ** alpha

    for attr in attributes:
        base_weight = attr['initial_weight']
        semantic_score = calculate_semantic_score(attr, attributes[0])
        # 按照文档公式计算权重
        adjusted_weight = (semantic_score ** alpha * base_weight) / denominator
        key = attr['value']
        if isinstance(key, list):
            key = tuple(key)  # 将列表转换为元组
        key = str(key)
        final_weights[key] = adjusted_weight

        # 记录语义强化信息
        if attr['name'] == '上下文信息':
            adjustment_log.append(f"语义强化: 上下文信息{attr['value']}+{adjusted_weight - base_weight:.2f}（CLIP相似度{semantic_score:.2f}）")
        else:
            adjustment_log.append(f"语义强化: {attr['value']}+{adjusted_weight - base_weight:.2f}（CLIP相似度{semantic_score:.2f}）")
    return final_weights, adjustment_log

def load_conflicts() -> Dict:
    """
    从conflicts.json文件加载显式冲突数据
    :return: 冲突数据字典
    """
    try:
        with open('conflicts.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {"conflict_cases": []}

def detect_conflict(attr1: Dict, attr2: Dict, conflicts: Dict) -> bool:
    """
    检测两个属性是否冲突
    :param attr1: 属性1
    :param attr2: 属性2
    :param conflicts: 从conflicts.json加载的冲突数据
    :return: 是否冲突
    """
    # 检查自定义的显式冲突
    combined_name = f"{attr1['value']}+{attr2['value']}"
    for case in conflicts['conflict_cases']:
        if combined_name in case['name']:
            return True

    # 隐式语义冲突（基于CLIP）
    similarity = calculate_semantic_score(attr1, attr2)
    return similarity < 0.3

def validate_request(data):
    """
    验证请求数据的有效性
    :param data: 请求数据
    :return: 验证结果和错误信息
    """
    if not data:
        return False, {
            "code": 40004,
            "message": "请求体不能为空",
            "solution": ["确保请求体包含有效的数据"]
        }
    if 'base_prompt' not in data or 'attributes' not in data:
        return False, {
            "code": 400,
            "message": "base_prompt和attributes是必填参数",
            "solution": ["在请求体中添加base_prompt和attributes字段"]
        }
    if not isinstance(data['attributes'], list) or len(data['attributes']) < 2:
        return False, {
            "code": 40002,
            "message": "attributes数组为空，至少提供2个属性对象",
            "solution": ["在attributes数组中添加至少2个属性对象"]
        }
    for attr in data['attributes']:
        if 'initial_weight' not in attr or not (0.0 <= attr['initial_weight'] <= 1.0):
            attr['initial_weight'] = 0.5
            return False, {
                "code": 40003,
                "message": "initial_weight超出[0.0,1.0]范围，使用0.5作为默认值并重试",
                "solution": ["将initial_weight的值设置在[0.0, 1.0]范围内"]
            }
        # 新增对type字段的验证
        if 'type' not in attr or attr['type'] not in ['text', 'image', 'vector']:
            return False, {
                "code": 40005,
                "message": "attribute的type字段必须为text、image或vector",
                "solution": ["将type字段的值设置为text、image或vector"]
            }
        # 对不同类型的数据进行校验
        if attr['type'] == 'text':
            if not isinstance(attr['value'], str):
                return False, {
                    "code": 40006,
                    "message": "text类型的value必须为字符串",
                    "solution": ["确保text类型的value为字符串"]
                }
        elif attr['type'] == 'image':
            if 'base64' not in attr['value']:
                return False, {
                    "code": 40007,
                    "message": "image类型的value必须包含base64字段",
                    "solution": ["确保image类型的value包含base64字段"]
                }
        elif attr['type'] == 'vector':
            if not isinstance(attr['value'], list):
                return False, {
                    "code": 40008,
                    "message": "vector类型的value必须为列表",
                    "solution": ["确保vector类型的value为列表"]
                }
    return True, None

def handle_conflicts(data, conflicts, fallback_strategy, temperature):
    """
    处理属性冲突
    :param data: 请求数据
    :param conflicts: 冲突数据
    :param fallback_strategy: 回退策略
    :param temperature: 温度参数
    :return: 最终权重和冲突报告
    """
    conflict_detected = False
    potential_conflicts = []
    for i in range(len(data['attributes'])):
        for j in range(i + 1, len(data['attributes'])):
            if detect_conflict(data['attributes'][i], data['attributes'][j], conflicts):
                conflict_detected = True
                potential_conflicts.append([data['attributes'][i]['value'], data['attributes'][j]['value']])

    if conflict_detected:
        if fallback_strategy =='strict':
            return None, {
                "code": 42201,
                "message": "显式冲突（如同时包含\"白天\"和\"夜晚\"）",
                "solution": ["移除冲突属性或切换降级策略"]
            }, []
        # balanced策略，重置为平均权重
        elif fallback_strategy == 'balanced':
            num_attrs = len(data['attributes'])
            avg_weight = 1.0 / num_attrs
            final_weights = {str(attr['value']): avg_weight for attr in data['attributes']}
            adjustment_log = [f"属性 {str(attr['value'])}: 因冲突采用平衡策略，权重调整为 {avg_weight}" for attr in
                              data['attributes']]
        else:
            # creative策略
            final_weights, adjustment_log = adjust_weights(data['attributes'], temperature)
    else:
        final_weights, adjustment_log = adjust_weights(data['attributes'], temperature)
        # 记录冲突检测信息
        adjustment_log.append("冲突检测: 无显式冲突")

    # 检查权重总和是否超出阈值
    weight_sum = sum(final_weights.values())
    if weight_sum > 1.5:
        return None, {
            "code": 42202,
            "message": "权重总和超出阈值（Σweights>1.5）",
            "solution": ["降低初始权重或启用自动归一化"]
        }, adjustment_log

    conflict_report = {
        "detected": conflict_detected,
        "potential_conflicts": potential_conflicts
    }
    return final_weights, conflict_report, adjustment_log

# 模拟请求签名校验
def verify_signature(request):
    api_key = "your_api_key"  # 替换为实际的API密钥
    timestamp = request.headers.get('X-Timestamp')
    signature = request.headers.get('X-Signature')
    if not timestamp or not signature:
        return False
    data = request.get_data(as_text=True)
    message = f"{data}{timestamp}{api_key}"
    calculated_signature = hashlib.sha256(message.encode()).hexdigest()
    return calculated_signature == signature

# 频率限制
redis_client = redis.Redis(host='localhost', port=6379, db=0)
def rate_limit(request):
    client_ip = request.remote_addr
    key = f"rate_limit:{client_ip}"
    current_count = redis_client.incr(key)
    if current_count == 1:
        redis_client.expire(key, 60)  # 限制每分钟请求次数
    if current_count > 50:  # 每分钟最多50次请求
        return False
    return True

@app.route('/api/v5/weight/calculate', methods=['POST'])
def calculate_weights():
    start_time = time.time()
    max_gpu_usage = 0  # 初始化最大 GPU 使用率为 0

    # 开始监控 GPU 使用率
    if device == "cuda":
        def monitor_gpu():
            nonlocal max_gpu_usage
            while True:
                gpus = GPUtil.getGPUs()
                for gpu in gpus:
                    gpu_usage = gpu.load * 100
                    if gpu_usage > max_gpu_usage:
                        max_gpu_usage = gpu_usage
                time.sleep(0.1)  # 每 0.1 秒检查一次

        monitor_thread = threading.Thread(target=monitor_gpu)
        monitor_thread.daemon = True
        monitor_thread.start()

    try:
        # 模拟请求签名校验
        '''
        if not verify_signature(request):
            return jsonify({
                "code": 40101,
                "message": "请求签名校验失败",
                "solution": ["检查请求签名和时间戳是否正确"]
            }), 401
        '''
        # 模拟频率限制
        if not rate_limit(request):
            return jsonify({
                "code": 42901,
                "message": "请求频率超出限制",
                "solution": ["降低请求频率"]
            }), 429

        # 检查请求头
        if 'Content-Type' not in request.headers or request.headers['Content-Type'] != 'application/json':
            return jsonify({
                "code": 400,
                "message": "Content-Type必须为application/json",
                "solution": ["在请求头中添加Content-Type字段并设置值为application/json"]
            }), 400
        if 'X-Api-Key' not in request.headers:
            return jsonify({
                "code": 400,
                "message": "缺少X-Api-Key",
                "solution": ["在请求头中添加X-Api-Key字段并设置有效的值"]
            }), 400

        data = request.get_json()
        if data is None:
            return jsonify({
                "code": 400,
                "message": "请求体不是有效的JSON格式",
                "solution": ["检查请求体的JSON格式是否正确"]
            }), 400

        # 验证请求数据
        valid, error = validate_request(data)
        if not valid:
            return jsonify(error), 400

        temperature = data.get('temperature', 1.2)
        if temperature < 0.1 or temperature > 5.0:
            temperature = 1.2
            return jsonify({
                "code": 40004,
                "message": "temperature值超出范围[0.0,5.0], 调整至范围内或使用默认值",
                "solution": ["将temperature的值设置在[0.1, 5.0]范围内"]
            }), 400
        if 'fallback_strategy' not in data:
            data["fallback_strategy"] = "balanced"
        fallback_strategy = data.get('fallback_strategy', 'balanced')
        if fallback_strategy not in ['strict', 'balanced', 'creative']:
            fallback_strategy = 'balanced'

        debug_mode = data.get('debug_mode', False)

        # 加载冲突数据
        conflicts = load_conflicts()

        # 获取上下文特征向量
        context_vector = encode_context()
        context_list = context_vector.tolist()

        # 将上下文信息作为新的属性添加到 attributes 中
        context_attribute = {
            "name": "上下文信息",
            "type": "vector",
            "value": context_list,
            "initial_weight": 0.1,
            "constraints": {
                "min_weight": 0.0,
                "max_weight": 1.0,
            }
        }

        #data['attributes'].append(context_attribute)

        # 处理冲突
        final_weights, conflict_report, adjustment_log = handle_conflicts(data, conflicts, fallback_strategy,
                                                                          temperature)
        if final_weights is None:
            return jsonify(conflict_report), 422

        end_time = time.time()
        processing_time_ms = int((end_time - start_time) * 1000)

        if device == "cuda":
            gpu_memory = torch.cuda.memory_allocated()
            gpu_max_memory = torch.cuda.max_memory_allocated()
            if gpu_memory >= gpu_max_memory * 0.9:  # 假设90%为内存不足阈值
                return jsonify({
                    "code": 50001,
                    "message": "GPU内存不足，减少单次请求属性数量",
                    "solution": ["减少单次请求中attributes数组的属性数量"]
                }), 500
        if device == "cuda":
            monitor_thread.join(timeout=1)

        response = {
            "code": 200,
            "data": {
                "final_weights": final_weights,
                "conflict_report": conflict_report,
                "adjustment_log": adjustment_log
            }
        }
        if debug_mode:
            response["debug_info"] = {
                "processing_time_ms": processing_time_ms,
                "model_version": "v2.1",
                "gpu_utilization": max_gpu_usage,
                "device_type": get_device_type(),
                "geo_location": get_geo_location(request.remote_addr)
            }
        return jsonify(response), 200
    except json.JSONDecodeError:
        return jsonify({
            "code": 40001,
            "message": "JSON格式错误",
            "solution": ["检查括号闭合或逗号分隔"]
        }), 400
    except torch.cuda.OutOfMemoryError:
        return jsonify({
            "code": 50001,
            "message": "GPU内存不足，减少单次请求属性数量",
            "solution": ["减少单次请求中attributes数组的属性数量"]
        }), 500
    except TimeoutError:
        return jsonify({
            "code": 50002,
            "message": "模型加载超时，等待1分钟后重试",
            "solution": ["等待1分钟后重新发起请求"]
        }), 500
    except Exception as e:
        print(f"系统级错误: {e}")
        if "library not found" in str(e):
            return jsonify({
                "code": 50003,
                "message": "动态库链接失败，检查CUDA版本是否兼容",
                "solution": ["检查CUDA版本是否与当前环境兼容"]
            }), 500
        return jsonify({
            "code": 500,
            "message": "系统级错误，未知原因",
            "error": str(e),
            "solution": ["检查服务器日志以获取更多详细信息"]
        }), 500

if __name__ == '__main__':
    app.run(debug=True)
    