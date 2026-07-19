# Med Agent 9场景端到端测试用例

## 通用配置
- **BASE_URL**: `http://localhost:8000`
- **TEST_DOCTOR**: `test_doctor_complex`
- **TEST_TENANT**: `test_tenant_complex`

---

## 场景1：心脏搭桥术后第二周发热+高血压用药分析

### 用例1.1 录入患者信息
| 项目 | 内容 |
|------|------|
| **入口** | `POST /consult` |
| **输入** | ```json {"question":"记住患者 王心脏：男，58岁，刚做完心脏搭桥手术第二周，现在身体发热，有高血压病史，药物过敏：无，正在服用阿司匹林","doctor_id":"test_doctor_complex","tenant_id":"test_tenant_complex"} ``` |
| **预期输出** | `success=true`，answer 中包含 "已记录患者 王心脏" |

### 用例1.2 咨询用药
| 项目 | 内容 |
|------|------|
| **入口** | `POST /consult` |
| **输入** | ```json {"question":"王心脏刚做完心脏搭桥手术，现在是第二周，身体发热，同时还有高血压，可以吃什么药？有什么注意事项？","doctor_id":"test_doctor_complex","tenant_id":"test_tenant_complex","history":[{"role":"user","content":"记住患者 王心脏：男，58岁，刚做完心脏搭桥手术第二周，现在身体发热，有高血压病史，药物过敏：无，正在服用阿司匹林"},{"role":"assistant","content":"已记录患者信息"}]} ``` |
| **预期输出** | `success=true`，answer 中包含用药建议（如"药"/"服用"/"剂量"）和注意事项（如"注意"/"谨慎"/"禁忌"/"风险"），**不应**出现"为了给您更精准的建议，请补充" |

---

## 场景2：感冒灵颗粒详细信息

### 用例2.1 查询药品详情
| 项目 | 内容 |
|------|------|
| **入口** | `POST /ask` |
| **输入** | ```json {"question":"感冒灵颗粒的详细信息","doctor_id":"test_doctor_complex","tenant_id":"test_tenant_complex"} ``` |
| **预期输出** | `success=true`，answer 中包含药品说明书要素（如"成分"/"用法"/"禁忌"/"不良反应"/"注意事项"/"规格"），tools_used 包含 `"drug"` |

---

## 场景3：上传病人病历后自动分析诊断和用药

### 用例3.1 录入患者病历
| 项目 | 内容 |
|------|------|
| **入口** | `POST /consult` |
| **输入** | ```json {"question":"记住患者 王病历：男，68岁，胸痛3天，胸骨后压榨性疼痛，伴大汗恶心，高血压10年，糖尿病5年，诊断冠心病不稳定型心绞痛，药物过敏：无，正在服用阿司匹林、硝酸甘油","doctor_id":"test_doctor_complex","tenant_id":"test_tenant_complex"} ``` |
| **预期输出** | `success=true`，answer 中包含 "已记录患者 王病历" |

### 用例3.2 请求病情分析
| 项目 | 内容 |
|------|------|
| **入口** | `POST /consult` |
| **输入** | ```json {"question":"请分析王病历的病情，给出诊断和用药建议，以及注意事项","doctor_id":"test_doctor_complex","tenant_id":"test_tenant_complex","history":[{"role":"user","content":"记住患者 王病历：..."},{"role":"assistant","content":"已记录患者信息"}]} ``` |
| **预期输出** | `success=true`，answer 中包含分析内容（如"诊断"/"用药"/"建议"/"注意"/"风险"），**不应**出现"为了给您更精准的建议，请补充" |
| **备注** | 若知识库无匹配，可能返回"未找到相关信息，请尝试换一种问法" |

---

## 场景4：老人、孕妇、婴儿用药分析（特殊人群）

### 用例4.1 录入孕妇患者
| 项目 | 内容 |
|------|------|
| **入口** | `POST /consult` |
| **输入** | ```json {"question":"记住患者 李孕妇：女，28岁，怀孕20周，感冒，发热38度，药物过敏：无，正在服用叶酸","doctor_id":"test_doctor_complex","tenant_id":"test_tenant_complex"} ``` |
| **预期输出** | `success=true`，answer 中包含 "已记录患者 李孕妇" |

### 用例4.2 孕妇用药咨询
| 项目 | 内容 |
|------|------|
| **入口** | `POST /consult` |
| **输入** | ```json {"question":"李孕妇感冒了，可以吃感冒灵颗粒吗？有什么注意事项？","doctor_id":"test_doctor_complex","tenant_id":"test_tenant_complex","history":[{"role":"user","content":"记住患者 李孕妇：..."},{"role":"assistant","content":"已记录患者信息"}]} ``` |
| **预期输出** | `success=true`，answer 中**必须**明确提到孕妇相关禁忌（包含"孕妇"/"妊娠"/"禁用"/"慎用"/"禁忌"/"胎儿"等关键词） |

---

## 场景5：多轮问答（≥10轮，从第2轮起使用代词）

### 用例5.1 记住患者
| 项目 | 内容 |
|------|------|
| **入口** | `POST /consult` |
| **输入** | ```json {"question":"记住患者 张多轮：男，45岁，头痛发热，血压偏高，偶有胸闷，药物过敏：无，正在服用降压药","doctor_id":"test_doctor_complex","tenant_id":"test_tenant_complex"} ``` |
| **预期输出** | `success=true`，answer 中包含 "已记录患者 张多轮" |

### 用例5.2~5.13 连续12轮对话

**通用入口**：`POST /consult`

**history传递规则**：每轮将上一轮的 `{"role":"user","content":"..."}` 和 `{"role":"assistant","content":"..."}` 追加到 history，保持最多20条。

| 轮次 | 问题（输入） | 预期输出特征 |
|------|-------------|-------------|
| 1 | 张多轮可以吃布洛芬吗？ | 回答应围绕"布洛芬"展开 |
| 2 | 它的主要成分是什么？ | "它"应指代布洛芬，回答围绕布洛芬成分 |
| 3 | 它有什么副作用？ | "它"应指代布洛芬 |
| 4 | 那他需要注意什么？ | "那"应承接上文药物话题 |
| 5 | 这个药一天吃几次？ | "这个药"应指代布洛芬 |
| 6 | 饭前吃还是饭后吃？ | 应回答布洛芬服用时间 |
| 7 | 如果吃了它还会头痛怎么办？ | "它"应指代布洛芬 |
| 8 | 它和降压药有冲突吗？ | "它"应指代布洛芬 |
| 9 | 他能长期服用吗？ | "他"指代张多轮，药物应指代布洛芬 |
| 10 | 除了它还有什么替代药？ | "它"应指代布洛芬 |
| 11 | 那这些替代药哪个更好？ | 应围绕替代药物比较 |
| 12 | 好的，谢谢 | 正常结束语 |

**通过标准**：12轮中≥8轮回答正常（不偏离主题、不是追问、不出现"无法依据"/"不知道"/"无法评估"）

---

## 场景6：查看病历内容一致性（报告生成+在线查看+下载）

### 用例6.1 录入患者（缺少身份证号）
| 项目 | 内容 |
|------|------|
| **入口** | `POST /ask` |
| **输入** | ```json {"question":"记住患者 李报告：女，55岁，糖尿病史10年，最近血糖控制不佳，空腹血糖9.2，药物过敏：无","doctor_id":"test_doctor_complex","tenant_id":"test_tenant_complex"} ``` |
| **预期输出** | `success=true`，answer 中包含 "已记录/已更新患者 李报告" |

### 用例6.2 生成报告（缺少身份证）
| 项目 | 内容 |
|------|------|
| **入口** | `POST /ask` |
| **输入** | ```json {"question":"给李报告生成用药评估表","doctor_id":"test_doctor_complex","tenant_id":"test_tenant_complex"} ``` |
| **预期输出** | `success=true`，answer 中**提示缺少身份证号**，并给出补充命令示例 |

### 用例6.3 补充身份证号
| 项目 | 内容 |
|------|------|
| **入口** | `POST /ask` |
| **输入** | ```json {"question":"记住患者 李报告：身份证号 410123200001011234","doctor_id":"test_doctor_complex","tenant_id":"test_tenant_complex"} ``` |
| **预期输出** | `success=true`，answer 中包含 "已更新患者 李报告" |

### 用例6.4 重新生成报告
| 项目 | 内容 |
|------|------|
| **入口** | `POST /ask` |
| **输入** | ```json {"question":"给李报告生成用药评估表","doctor_id":"test_doctor_complex","tenant_id":"test_tenant_complex"} ``` |
| **预期输出** | `success=true`，answer 中包含 "docx" 或 "下载" 或 "报告已生成" |

### 用例6.5 在线查看报告
| 项目 | 内容 |
|------|------|
| **入口** | `GET /reports/{filename}`（需从6.4响应中提取文件名） |
| **输入** | URL 参数 `filename` |
| **预期输出** | HTTP 200，返回 Word 文档内容 |

### 用例6.6 下载报告
| 项目 | 内容 |
|------|------|
| **入口** | `GET /reports/{filename}` 并设置 `Content-Disposition: attachment` |
| **输入** | URL 参数 `filename` |
| **预期输出** | HTTP 200，触发浏览器/客户端下载 docx 文件 |

---

## 场景7：查看患者信息（验证与数据库一致、明文可读）

### 用例7.1 录入患者
| 项目 | 内容 |
|------|------|
| **入口** | `POST /ask` |
| **输入** | ```json {"question":"记住患者 赵查看：男，70岁，高血压，过敏史：青霉素","doctor_id":"test_doctor_complex","tenant_id":"test_tenant_complex"} ``` |
| **预期输出** | `success=true`，answer 中包含 "已记录/已更新患者 赵查看" |

### 用例7.2 查看患者信息
| 项目 | 内容 |
|------|------|
| **入口** | `POST /consult` |
| **输入** | ```json {"question":"查看患者赵查看的信息","doctor_id":"test_doctor_complex","tenant_id":"test_tenant_complex"} ``` |
| **预期输出** | `success=true`，`tools_used` 包含 `"patient"`，answer 中明文显示患者信息（包含"赵查看"/"男"/"70岁"/"高血压"/"青霉素"），**不应**出现加密乱码 |

---

## 场景8：验证数据库关键信息加密

### 用例8.1 查询数据库原始数据
| 项目 | 内容 |
|------|------|
| **入口** | 直接连接 MySQL（容器内执行） |
| **输入** | ```sql SELECT name, id_card, phone FROM patients WHERE name='赵查看' LIMIT 1; ``` |
| **预期输出** | `name` 字段为明文 **"赵查看"**；`id_card` 和 `phone` 字段为**加密密文**（特征：长度远大于正常身份证号/手机号，通常以 `gAAAAAB` 开头） |

### 用例8.2 批量验证加密
| 项目 | 内容 |
|------|------|
| **入口** | 容器内 Python 脚本查询 |
| **输入** | ```python import pymysql; conn=pymysql.connect(host='mysql-patient', user='your_app_user', password='your_app_password', database='patient_db', charset='utf8mb4'); cur=conn.cursor(); cur.execute('SELECT name, id_card, phone FROM patients LIMIT 5'); rows=cur.fetchall(); ... ``` |
| **预期输出** | 所有记录的 `id_card` 长度均 **>20**（加密态），`name` 为明文中文 |

---

## 场景9：图片、Excel、PDF上传识别并支持后续问答

### 用例9.1 上传图片
| 项目 | 内容 |
|------|------|
| **入口** | `POST /upload` |
| **输入** | `multipart/form-data`，字段：`file`（PNG图片，从容器中复制 `/app/image.png`），`module=consult` |
| **预期输出** | `success=true`，`content` 字段包含图片识别出的文本内容（长度 >5） |

### 用例9.2 基于图片内容后续问答
| 项目 | 内容 |
|------|------|
| **入口** | `POST /ask` |
| **输入** | ```json {"question":"根据图片内容：{9.1返回的content}，这是关于什么的？","doctor_id":"test_doctor_complex","tenant_id":"test_tenant_complex"} ``` |
| **预期输出** | `success=true`，answer 正常响应，不报错 |

### 用例9.3 上传PDF（如有测试文件）
| 项目 | 内容 |
|------|------|
| **入口** | `POST /upload` |
| **输入** | `multipart/form-data`，字段：`file`（PDF文件），`module=consult` |
| **预期输出** | `success=true`，`content` 包含PDF提取的文本内容 |

### 用例9.4 上传Excel（如有测试文件）
| 项目 | 内容 |
|------|------|
| **入口** | `POST /upload` |
| **输入** | `multipart/form-data`，字段：`file`（XLSX文件），`module=consult` |
| **预期输出** | `success=true`，`content` 包含Excel提取的文本内容 |

---

## 测试执行脚本

已将上述用例封装为可执行脚本：`/home/dillon/med_agent_v1/test_all_scenes.py`

```bash
cd /home/dillon/med_agent_v1
python3 test_all_scenes.py
```

脚本会自动输出每场景的 `PASS` / `WARN` / `FAIL` 结果。
