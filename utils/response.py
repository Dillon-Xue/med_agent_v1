import re
import time
import uuid


def mask_sensitive(text: str) -> str:
    """对日志中的敏感信息进行脱敏"""
    if not isinstance(text, str):
        return text
    # 身份证号（18位）
    text = re.sub(r'(\d{6})\d{8}(\d{4})', r'\1********\2', text)
    # 手机号（11位）
    text = re.sub(r'(1[3-9])(\d{4})(\d{4})', r'\1****\3', text)
    return text


def mask_dict_sensitive(data: dict) -> dict:
    """递归脱敏字典中的所有字符串值"""
    if not isinstance(data, dict):
        return data
    result = {}
    for key, value in data.items():
        if isinstance(value, str):
            result[key] = mask_sensitive(value)
        elif isinstance(value, dict):
            result[key] = mask_dict_sensitive(value)
        elif isinstance(value, list):
            result[key] = [
                mask_dict_sensitive(item) if isinstance(item, dict) 
                else mask_sensitive(str(item)) if isinstance(item, str) 
                else item 
                for item in value
            ]
        else:
            result[key] = value
    return result


def build_response(answer, source, debug=None, trace=None, success=True, mask_log=True):
    if mask_log and debug:
        debug = mask_dict_sensitive(debug)
    
    if mask_log and answer and isinstance(answer, str):
        answer = mask_sensitive(answer)
    
    return {
        "success": success,
        "id": str(uuid.uuid4()),
        "source": source,
        "answer": answer,
        "debug": debug or {},
        "trace": trace or [],
        "timestamp": time.time()
    }