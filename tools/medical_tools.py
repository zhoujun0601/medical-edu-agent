"""
医学工具定义 - 供 LLM function calling 使用
"""
from typing import Dict, Any, List
import json

# ======================================================
#  工具定义（OpenAI function calling 格式）
# ======================================================

MEDICAL_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_drug_information",
            "description": "查询药物的详细信息，包括适应症、用法用量、禁忌症、不良反应和药物相互作用",
            "parameters": {
                "type": "object",
                "properties": {
                    "drug_name": {
                        "type": "string",
                        "description": "药物通用名称（中文或英文），如：阿莫西林、Amoxicillin"
                    },
                    "query_type": {
                        "type": "string",
                        "enum": ["full", "dosage", "interactions", "contraindications", "adverse_effects"],
                        "description": "查询类型：full=完整信息，dosage=用法用量，interactions=药物相互作用，contraindications=禁忌症，adverse_effects=不良反应"
                    }
                },
                "required": ["drug_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_clinical_score",
            "description": "计算临床评分量表，如 Wells 评分（DVT/PE）、CURB-65（肺炎）、Glasgow 昏迷评分、Child-Pugh 评分等",
            "parameters": {
                "type": "object",
                "properties": {
                    "score_name": {
                        "type": "string",
                        "enum": ["wells_dvt", "wells_pe", "curb65", "glasgow", "child_pugh", "apache2", "sofa", "chad2ds2", "has_bled"],
                        "description": "评分量表名称"
                    },
                    "parameters": {
                        "type": "object",
                        "description": "评分参数，格式因量表而异"
                    }
                },
                "required": ["score_name", "parameters"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_clinical_guideline",
            "description": "检索相关临床指南和诊疗规范，提供最新的循证医学建议",
            "parameters": {
                "type": "object",
                "properties": {
                    "condition": {
                        "type": "string",
                        "description": "疾病或临床问题，如：2型糖尿病、急性心肌梗死"
                    },
                    "organization": {
                        "type": "string",
                        "enum": ["chinese", "aha_acc", "esc", "who", "nice", "any"],
                        "description": "指南来源：chinese=中国指南，aha_acc=美国心脏学会，esc=欧洲心脏学会，who=世界卫生组织，any=任意"
                    }
                },
                "required": ["condition"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_exam_question",
            "description": "生成医学考试模拟题，适用于执业医师考试、住院医师规培考核等",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject": {
                        "type": "string",
                        "enum": ["internal_medicine", "surgery", "obstetrics", "pediatrics", "neurology", "pharmacology", "pathology", "physiology"],
                        "description": "考试科目"
                    },
                    "question_type": {
                        "type": "string",
                        "enum": ["single_choice", "multiple_choice", "case_analysis"],
                        "description": "题型"
                    },
                    "difficulty": {
                        "type": "string",
                        "enum": ["basic", "intermediate", "advanced"],
                        "description": "难度级别"
                    },
                    "count": {
                        "type": "integer",
                        "description": "生成题目数量（1-10）",
                        "minimum": 1,
                        "maximum": 10
                    }
                },
                "required": ["subject"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "interpret_lab_result",
            "description": "解读实验室检查结果，判断是否异常，提供临床意义分析",
            "parameters": {
                "type": "object",
                "properties": {
                    "test_name": {
                        "type": "string",
                        "description": "检查项目名称，如：血红蛋白、肌酐、ALT"
                    },
                    "value": {
                        "type": "number",
                        "description": "检测数值"
                    },
                    "unit": {
                        "type": "string",
                        "description": "单位，如：g/L、μmol/L、U/L"
                    },
                    "patient_info": {
                        "type": "object",
                        "description": "患者基本信息（可选）",
                        "properties": {
                            "age": {"type": "integer"},
                            "gender": {"type": "string", "enum": ["male", "female"]},
                            "pregnant": {"type": "boolean"}
                        }
                    }
                },
                "required": ["test_name", "value", "unit"]
            }
        }
    }
]


# ======================================================
#  工具执行器（模拟数据 + 简单逻辑）
# ======================================================

class MedicalToolExecutor:
    
    @staticmethod
    def execute(tool_name: str, arguments: Dict[str, Any]) -> str:
        """执行工具调用"""
        handlers = {
            "get_drug_information": MedicalToolExecutor._get_drug_info,
            "calculate_clinical_score": MedicalToolExecutor._calculate_score,
            "search_clinical_guideline": MedicalToolExecutor._search_guideline,
            "generate_exam_question": MedicalToolExecutor._generate_question,
            "interpret_lab_result": MedicalToolExecutor._interpret_lab,
        }
        handler = handlers.get(tool_name)
        if not handler:
            return json.dumps({"error": f"未知工具: {tool_name}"}, ensure_ascii=False)
        try:
            return handler(**arguments)
        except Exception as e:
            return json.dumps({"error": str(e)}, ensure_ascii=False)

    @staticmethod
    def _get_drug_info(drug_name: str, query_type: str = "full") -> str:
        """药物信息查询（示例数据结构，实际应接数据库）"""
        result = {
            "drug_name": drug_name,
            "query_type": query_type,
            "note": f"以下是 {drug_name} 的基本信息框架，实际系统中应接入专业药物数据库（如 CFDA、DrugBank）",
            "instruction": f"请基于您的医学知识，结合此查询意图（{query_type}），为学生提供 {drug_name} 的详细教学信息"
        }
        return json.dumps(result, ensure_ascii=False, indent=2)

    @staticmethod
    def _calculate_score(score_name: str, parameters: Dict) -> str:
        """临床评分计算"""
        score_map = {
            "wells_dvt": {
                "name": "Wells DVT 评分",
                "description": "深静脉血栓形成预测评分",
                "items": ["活动性癌症(+1)", "瘫痪/近期石膏固定(+1)", "近期卧床>3天或12周内大手术(+1)", "沿深静脉走行局部压痛(+1)", "整条腿肿胀(+1)", "小腿肿胀>3cm(+1)", "凹陷性水肿(+1)", "浅静脉侧支循环(+1)", "既往DVT(+1)", "其他诊断可能性≥DVT(-2)"],
                "interpretation": "评分≥2: 高可能性; 0-1: 低可能性"
            },
            "curb65": {
                "name": "CURB-65 评分（社区获得性肺炎）",
                "items": ["意识障碍(+1)", "尿素氮>7mmol/L(+1)", "呼吸频率≥30次/分(+1)", "低血压: SBP<90或DBP≤60mmHg(+1)", "年龄≥65岁(+1)"],
                "interpretation": "0-1分: 低危(门诊); 2分: 中危(住院); ≥3分: 高危(ICU考虑)"
            },
        }
        info = score_map.get(score_name, {"name": score_name, "note": "请参考最新临床指南"})
        result = {
            "score_name": score_name,
            "score_info": info,
            "received_parameters": parameters,
            "note": "请基于以上评分标准和接收到的参数，为学生计算具体评分并解释临床意义"
        }
        return json.dumps(result, ensure_ascii=False, indent=2)

    @staticmethod
    def _search_guideline(condition: str, organization: str = "any") -> str:
        result = {
            "condition": condition,
            "organization": organization,
            "note": f"正在为 [{condition}] 检索相关指南（组织偏好: {organization}）",
            "instruction": "请基于您的医学知识，提供该疾病的最新主要诊疗指南要点，注明指南来源和年份"
        }
        return json.dumps(result, ensure_ascii=False, indent=2)

    @staticmethod
    def _generate_question(subject: str, question_type: str = "single_choice", 
                           difficulty: str = "intermediate", count: int = 1) -> str:
        subject_names = {
            "internal_medicine": "内科学", "surgery": "外科学",
            "obstetrics": "妇产科学", "pediatrics": "儿科学",
            "neurology": "神经内科学", "pharmacology": "药理学",
            "pathology": "病理学", "physiology": "生理学"
        }
        result = {
            "subject": subject_names.get(subject, subject),
            "question_type": question_type,
            "difficulty": difficulty,
            "count": count,
            "instruction": f"请为{subject_names.get(subject, subject)}生成{count}道{'单选题' if question_type == 'single_choice' else '多选题' if question_type == 'multiple_choice' else '案例分析题'}（{difficulty}难度），每题包含题干、选项、正确答案和详细解析"
        }
        return json.dumps(result, ensure_ascii=False, indent=2)

    @staticmethod
    def _interpret_lab(test_name: str, value: float, unit: str, patient_info: Dict = None) -> str:
        result = {
            "test_name": test_name,
            "value": value,
            "unit": unit,
            "patient_info": patient_info or {},
            "instruction": f"请解读 {test_name} = {value} {unit} 的临床意义，说明参考范围、是否异常、可能原因及临床建议"
        }
        return json.dumps(result, ensure_ascii=False, indent=2)
