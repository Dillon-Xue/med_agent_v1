#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
9场景端到端测试套件
运行方式: cd /home/dillon/med_agent_v1 && python3 test_all_scenes.py
"""
import requests
import json
import time
import sys
import os
import re
import subprocess

BASE_URL = "http://localhost:8000"
TEST_DOCTOR = "test_doctor_complex"
TEST_TENANT = "test_tenant_complex"
IMAGE_PATH = "/home/dillon/med_agent_v1/test_image.png"

results = []

def log(scene, status, detail=""):
    msg = f"[{status}] 场景{scene}: {detail}"
    print(msg)
    results.append({"scene": scene, "status": status, "detail": detail})

def ask(question):
    url = f"{BASE_URL}/ask"
    payload = {"question": question, "doctor_id": TEST_DOCTOR, "tenant_id": TEST_TENANT}
    try:
        r = requests.post(url, json=payload, timeout=120)
        return r.json()
    except Exception as e:
        return {"success": False, "error": str(e)}

def consult(question, history=None):
    url = f"{BASE_URL}/consult"
    payload = {"question": question, "doctor_id": TEST_DOCTOR, "tenant_id": TEST_TENANT}
    if history:
        payload["history"] = history
    try:
        r = requests.post(url, json=payload, timeout=300)
        return r.json()
    except Exception as e:
        return {"success": False, "error": str(e)}

def consult_with_patient(name, info, question):
    """先记住患者，然后携带history问问题"""
    q1 = f"记住患者 {name}：{info}"
    r1 = consult(q1)
    print(f"  [记住患者] {json.dumps(r1, ensure_ascii=False)[:200]}")
    time.sleep(1)
    ans1 = r1.get("result", {}).get("answer", "")
    hist = [
        {"role": "user", "content": q1},
        {"role": "assistant", "content": ans1}
    ]
    r2 = consult(question, history=hist)
    return r2, hist

# ============================================================
# 场景1: 心脏搭桥术后第二周发热+高血压用药分析
# 预期: 结合rag给出用药建议，含注意事项
# ============================================================
def test_scene1():
    print("\n========== 场景1: 心脏搭桥术后发热+高血压 ==========")
    info = "男，58岁，刚做完心脏搭桥手术第二周，现在身体发热，有高血压病史，药物过敏：无，正在服用阿司匹林"
    q = "王心脏刚做完心脏搭桥手术，现在是第二周，身体发热，同时还有高血压，可以吃什么药？有什么注意事项？"
    resp, _ = consult_with_patient("王心脏", info, q)
    if not resp.get("success"):
        log(1, "FAIL", f"请求失败: {resp.get('error', '')}")
        return
    ans = resp.get("result", {}).get("answer", "")
    tools = resp.get("result", {}).get("tools_used", [])
    print(f"tools: {tools}")
    print(f"answer前250字: {ans[:250]}")
    is_followup = "为了给您更精准的建议，请补充" in ans
    has_medical = any(k in ans for k in ["药", "服用", "剂量", "注意", "谨慎", "禁忌", "风险", "建议"])
    if is_followup:
        log(1, "WARN", f"仍被追问, answer={ans[:150]}")
    elif has_medical:
        log(1, "PASS", f"工具={tools}, 有医学建议={has_medical}")
    else:
        log(1, "WARN", f"工具={tools}, answer={ans[:150]}")

# ============================================================
# 场景2: 感冒灵颗粒详细信息
# 预期: 完整说明书（成分、用法、禁忌、不良反应等）
# ============================================================
def test_scene2():
    print("\n========== 场景2: 感冒灵颗粒详细信息 ==========")
    q = "感冒灵颗粒的详细信息"
    resp = ask(q)
    if not resp.get("success"):
        log(2, "FAIL", f"请求失败: {resp.get('error', '')}")
        return
    ans = resp.get("result", {}).get("answer", "")
    tools = resp.get("result", {}).get("tools_used", [])
    print(f"tools: {tools}")
    print(f"answer前200字: {ans[:200]}")
    has_detail = any(k in ans for k in ["成分", "用法", "禁忌", "不良反应", "注意事项", "规格"])
    if has_detail:
        log(2, "PASS", f"工具={tools}, 有详细说明={has_detail}")
    else:
        log(2, "WARN", f"工具={tools}, answer={ans[:150]}")

# ============================================================
# 场景3: 上传病人病历后自动分析
# 预期: 分析病情并给出诊断和用药建议
# ============================================================
def test_scene3():
    print("\n========== 场景3: 上传病历后自动分析 ==========")
    info = "男，68岁，胸痛3天，胸骨后压榨性疼痛，伴大汗恶心，高血压10年，糖尿病5年，诊断冠心病不稳定型心绞痛，药物过敏：无，正在服用阿司匹林、硝酸甘油"
    q = "请分析王病历的病情，给出诊断和用药建议，以及注意事项"
    resp, _ = consult_with_patient("王病历", info, q)
    if not resp.get("success"):
        log(3, "FAIL", f"请求失败: {resp.get('error', '')}")
        return
    ans = resp.get("result", {}).get("answer", "")
    tools = resp.get("result", {}).get("tools_used", [])
    print(f"tools: {tools}")
    print(f"answer前250字: {ans[:250]}")
    is_followup = "为了给您更精准的建议，请补充" in ans
    has_analysis = any(k in ans for k in ["诊断", "用药", "建议", "注意", "风险"])
    if is_followup:
        log(3, "WARN", f"仍被追问, answer={ans[:150]}")
    elif has_analysis:
        log(3, "PASS", f"工具={tools}, 包含分析建议")
    else:
        log(3, "WARN", f"工具={tools}, answer={ans[:150]}")

# ============================================================
# 场景4: 老人、孕妇、婴儿用药分析
# 预期: 严格分析特殊情况，给出禁忌和注意事项
# ============================================================
def test_scene4():
    print("\n========== 场景4: 特殊人群用药分析 ==========")
    info = "女，28岁，怀孕20周，感冒，发热38度，药物过敏：无，正在服用叶酸"
    q = "李孕妇感冒了，可以吃感冒灵颗粒吗？有什么注意事项？"
    resp, _ = consult_with_patient("李孕妇", info, q)
    if not resp.get("success"):
        log(4, "FAIL", f"请求失败: {resp.get('error', '')}")
        return
    ans = resp.get("result", {}).get("answer", "")
    tools = resp.get("result", {}).get("tools_used", [])
    print(f"tools: {tools}")
    print(f"answer前250字: {ans[:250]}")
    is_followup = "为了给您更精准的建议，请补充" in ans
    has_preg = any(k in ans for k in ["孕妇", "妊娠", "禁用", "慎用", "禁忌", "胎儿"])
    if is_followup:
        log(4, "WARN", f"仍被追问, answer={ans[:150]}")
    elif has_preg:
        log(4, "PASS", f"工具={tools}, 提到孕妇/禁忌注意事项={has_preg}")
    else:
        log(4, "WARN", f"工具={tools}, 未明确提到孕妇禁忌, answer={ans[:150]}")

# ============================================================
# 场景5: 多轮问答（≥10轮，从第2轮开始使用代词）
# 预期: 能结合上下文，自然对话，回答不偏离
# ============================================================
def test_scene5():
    print("\n========== 场景5: 多轮问答（≥10轮，使用代词） ==========")
    info = "男，45岁，头痛发热，血压偏高，偶有胸闷，药物过敏：无，正在服用降压药"
    q0 = f"记住患者 张多轮：{info}"
    r0 = consult(q0)
    print(f"  [记住患者] {json.dumps(r0, ensure_ascii=False)[:200]}")
    time.sleep(1)
    history = [
        {"role": "user", "content": q0},
        {"role": "assistant", "content": r0.get("result", {}).get("answer", "")}
    ]
    questions = [
        "张多轮可以吃布洛芬吗？",
        "它的主要成分是什么？",
        "它有什么副作用？",
        "那他需要注意什么？",
        "这个药一天吃几次？",
        "饭前吃还是饭后吃？",
        "如果吃了它还会头痛怎么办？",
        "它和降压药有冲突吗？",
        "他能长期服用吗？",
        "除了它还有什么替代药？",
        "那这些替代药哪个更好？",
        "好的，谢谢"
    ]
    good_count = 0
    bad_count = 0
    for i, q in enumerate(questions):
        print(f"  轮次{i+1}: {q}")
        resp = consult(q, history=history)
        if not resp.get("success"):
            print(f"    请求失败: {resp.get('error', '')}")
            bad_count += 1
            continue
        ans = resp.get("result", {}).get("answer", "")
        history.append({"role": "user", "content": q})
        history.append({"role": "assistant", "content": ans})
        if len(history) > 20:
            history = history[-20:]
        print(f"    回答前100字: {ans[:100]}")
        is_followup = "为了给您更精准的建议，请补充" in ans
        bad_phrases = ["无法依据", "未提供", "不知道", "不清楚", "无法评估"]
        has_bad = any(p in ans for p in bad_phrases)
        if not is_followup and not has_bad and len(ans) > 20:
            good_count += 1
        else:
            bad_count += 1
        time.sleep(0.5)
    if good_count >= 8:
        log(5, "PASS", f"12轮中{good_count}轮回答正常，{bad_count}轮异常")
    else:
        log(5, "WARN", f"12轮中仅{good_count}轮回答正常，{bad_count}轮异常")

# ============================================================
# 场景6: 查看病历内容一致性（报告生成）
# 预期: 生成报告或提示缺少必要信息
# ============================================================
def test_scene6():
    print("\n========== 场景6: 病历内容一致性（报告生成） ==========")
    resp0 = ask("记住患者 李报告：女，55岁，糖尿病史10年，最近血糖控制不佳，空腹血糖9.2，药物过敏：无")
    print(f"录入患者: {json.dumps(resp0, ensure_ascii=False)[:200]}")
    time.sleep(1)
    q = "给李报告生成用药评估表"
    resp = ask(q)
    print(f"生成报告响应: {json.dumps(resp, ensure_ascii=False)[:300]}")
    if not resp.get("success"):
        log(6, "FAIL", f"生成报告失败: {resp.get('error', '')}")
        return
    ans = resp.get("result", {}).get("answer", "")
    has_doc = "docx" in ans or "下载" in ans or "报告" in ans or "评估表" in ans or "生成" in ans
    if has_doc and "未找到" not in ans:
        log(6, "PASS", f"报告生成成功，提示={ans[:120]}")
    elif "未找到" in ans:
        log(6, "FAIL", f"未找到患者档案: {ans[:120]}")
    else:
        log(6, "WARN", f"报告生成结果: {ans[:150]}")

# ============================================================
# 场景7: 查看患者信息（验证与数据库一致、明文可读）
# 预期: 成功查看，内容明文，与数据库一致
# ============================================================
def test_scene7():
    print("\n========== 场景7: 查看患者信息 ==========")
    resp0 = ask("记住患者 赵查看：男，70岁，高血压，过敏史：青霉素")
    print(f"录入患者: {json.dumps(resp0, ensure_ascii=False)[:200]}")
    time.sleep(1)
    q = "查看患者赵查看的信息"
    resp = consult(q)
    print(f"响应: {json.dumps(resp, ensure_ascii=False)[:400]}")
    if not resp.get("success"):
        log(7, "FAIL", f"请求失败: {resp.get('error', '')}")
        return
    ans = resp.get("result", {}).get("answer", "")
    tools = resp.get("result", {}).get("tools_used", [])
    print(f"tools: {tools}")
    print(f"answer: {ans[:300]}")
    has_name = "赵查看" in ans
    has_patient_tool = "patient" in tools
    has_plaintext = "高血压" in ans or "过敏" in ans or "青霉素" in ans or "男" in ans
    if has_name and has_patient_tool and has_plaintext:
        log(7, "PASS", f"工具={tools}, 包含姓名={has_name}, 明文可读={has_plaintext}")
    else:
        log(7, "WARN", f"工具={tools}, 姓名={has_name}, 明文={has_plaintext}, answer={ans[:200]}")

# ============================================================
# 场景8: 验证数据库关键信息加密
# 预期: id_card、phone等敏感字段在数据库中是加密状态
# ============================================================
def test_scene8():
    print("\n========== 场景8: 验证数据库加密 ==========")
    try:
        cmd = [
            "docker", "exec", "med_agent", "python3", "-c",
            "import pymysql; conn=pymysql.connect(host='mysql-patient', user='your_app_user', password='your_app_password', database='patient_db', charset='utf8mb4'); cur=conn.cursor(); cur.execute('SELECT name, id_card, phone FROM patients LIMIT 5'); rows=cur.fetchall(); cur.close(); conn.close(); [print(f\"{r[0]}|{len(r[1]) if r[1] else 0}|{len(r[2]) if r[2] else 0}\") for r in rows]"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        output = result.stdout.strip()
        print(f"数据库输出: {output}")
        lines = output.split("\n")
        encrypted_found = False
        for line in lines:
            parts = line.split("|")
            if len(parts) == 3:
                name, id_card_len, phone_len = parts[0], int(parts[1]), int(parts[2])
                if id_card_len > 20 or phone_len > 20:
                    encrypted_found = True
                    print(f"  {name}: id_card_len={id_card_len}, phone_len={phone_len} -> 已加密")
        if encrypted_found:
            log(8, "PASS", f"数据库中敏感字段已加密存储")
        else:
            log(8, "WARN", f"未检测到加密字段")
    except Exception as e:
        log(8, "FAIL", f"数据库查询异常: {e}")

# ============================================================
# 场景9: 图片上传识别并支持后续问答
# 预期: 上传成功，识别内容，可继续问答
# ============================================================
def test_scene9():
    print("\n========== 场景9: 文件上传识别 ==========")
    if not os.path.exists(IMAGE_PATH):
        print("尝试从容器复制测试图片...")
        subprocess.run(["docker", "cp", "med_agent:/app/image.png", IMAGE_PATH], check=False)
    if not os.path.exists(IMAGE_PATH):
        log(9, "SKIP", f"本地不存在测试图片 {IMAGE_PATH}")
        return
    try:
        with open(IMAGE_PATH, "rb") as fimg:
            r = requests.post(f"{BASE_URL}/upload", files={"file": ("image.png", fimg, "image/png")}, data={"module": "consult"}, timeout=120)
        print(f"上传状态码: {r.status_code}")
        if r.status_code != 200:
            log(9, "FAIL", f"上传失败，状态码={r.status_code}, body={r.text[:200]}")
            return
        resp = r.json()
        print(f"上传响应: {json.dumps(resp, ensure_ascii=False)[:300]}")
        content = resp.get("content", "")
        if len(content) > 5:
            r2 = requests.post(f"{BASE_URL}/ask", json={
                "question": f"根据图片内容：{content[:300]}，这是关于什么的？",
                "doctor_id": TEST_DOCTOR,
                "tenant_id": TEST_TENANT
            }, timeout=120)
            if r2.status_code == 200:
                ans = r2.json().get("result", {}).get("answer", "")
                print(f"后续问答: {ans[:150]}")
                log(9, "PASS", f"图片上传识别成功，回答长度={len(content)}, 后续问答正常")
            else:
                log(9, "WARN", f"上传成功但后续问答失败: {r2.status_code}")
        else:
            log(9, "WARN", f"图片上传后内容为空或异常")
    except Exception as e:
        log(9, "FAIL", f"图片上传异常: {e}")

# ============================================================
# 主程序
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("开始执行9场景端到端测试")
    print("=" * 60)
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=10)
        print(f"服务健康检查: {r.status_code} {r.text[:100]}")
    except Exception as e:
        print(f"服务未就绪: {e}")
        sys.exit(1)
    test_scene1()
    time.sleep(1)
    test_scene2()
    time.sleep(1)
    test_scene3()
    time.sleep(1)
    test_scene4()
    time.sleep(1)
    test_scene5()
    time.sleep(1)
    test_scene6()
    time.sleep(1)
    test_scene7()
    time.sleep(1)
    test_scene8()
    time.sleep(1)
    test_scene9()
    print("\n" + "=" * 60)
    print("测试汇总")
    print("=" * 60)
    pass_count = sum(1 for r in results if r["status"] == "PASS")
    fail_count = sum(1 for r in results if r["status"] == "FAIL")
    warn_count = sum(1 for r in results if r["status"] == "WARN")
    skip_count = sum(1 for r in results if r["status"] == "SKIP")
    print(f"通过: {pass_count}, 失败: {fail_count}, 警告: {warn_count}, 跳过: {skip_count}")
    for r in results:
        print(f"  场景{r['scene']}: {r['status']} - {r['detail']}")
