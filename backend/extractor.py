"""
人物关系提取与聚合模块
包含 LLM Prompt 设计、并行提取、递归聚合
"""
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from chunker import preprocess_text, chunk_text_with_metadata
from llm_client import LLMClient

logger = logging.getLogger('silverfish.extractor')


GENERIC_PERSON_ROLES = {
    '人', '男人', '女人', '男子', '女子', '老人', '孩子', '儿童', '婴儿', '少年', '青年',
    '父亲', '母亲', '父母', '丈夫', '妻子', '儿子', '女儿', '兄弟', '姐妹', '亲戚', '家属',
    '朋友', '邻居', '客人', '陌生人', '路人', '居民', '村民', '市民', '群众', '众人',
    '士兵', '军人', '军官', '将军', '上校', '中校', '少校', '上尉', '中尉', '少尉', '司令',
    '总统', '官员', '长官', '警察', '卫兵', '守卫', '法官', '律师', '书记员', '公务员',
    '医生', '护士', '药剂师', '教师', '老师', '学生', '校长', '神甫', '牧师', '修女', '神父',
    '工人', '农民', '商人', '店主', '老板', '伙计', '仆人', '女仆', '佣人', '厨师', '车夫',
    '先生', '女士', '小姐', '夫人', '太太', '绅士', '贵妇', '寡妇', '少女', '妇女',
    '妓女', '情人', '未婚妻', '新娘', '新郎',
    '领导', '首领', '代表', '使者', '信使', '囚犯', '犯人', '死者', '病人', '疯子', '流浪汉',
}

GENERIC_PERSON_MODIFIER_RE = re.compile(
    r'^(?:(?:一|二|两|三|四|五|六|七|八|九|十|百|几|数|多|若干|一群|一批)'
    r'(?:个|名|位|群)?|某(?:个|名|位)?|这(?:个|名|位)?|那(?:个|名|位)?|'
    r'年轻的?|年老的?|陌生的?|当地的?|外地的?|普通的?|其他的?|其余的?|'
    r'共和国|政府|军队|保守党|自由党|学校|村里|镇上|城里的?)+$'
)

PERSON_TITLE_PREFIXES = (
    '共和国总统', '总统', '将军', '司令', '上校', '中校', '少校', '上尉', '中尉', '少尉',
    '神父', '神甫', '牧师', '修女', '医生', '律师', '法官', '市长', '校长', '教授', '博士',
    '先生', '女士', '小姐', '夫人', '太太', '堂',
)
PERSON_TITLE_SUFFIXES = tuple(
    sorted((title for title in PERSON_TITLE_PREFIXES if title != '共和国总统'), key=len, reverse=True)
)


def is_generic_person_name(value):
    """判断名称是否只是数量词/修饰语加职业、身份或亲属泛称。"""
    compact = re.sub(r'[\s“”"\'()（）【】\[\]，,。.!！?？:：;；·•・._—-]+', '', str(value or '')).strip()
    compact = compact.removesuffix('们')
    if not compact or compact in GENERIC_PERSON_ROLES:
        return True
    for role in sorted(GENERIC_PERSON_ROLES, key=len, reverse=True):
        if not compact.endswith(role):
            continue
        modifier = compact[:-len(role)]
        if modifier and GENERIC_PERSON_MODIFIER_RE.fullmatch(modifier):
            return True
    return False


def normalize_person_name(value):
    """移除具体姓名前后的职务/尊称，返回标准姓名；纯泛称仍交给过滤器处理。"""
    original = re.sub(r'\s+', ' ', str(value or '')).strip(' “ ”"\'()（）【】[]，,。.!！?？:：;；')
    if not original or is_generic_person_name(original):
        return ''
    canonical = original
    changed = True
    while changed:
        changed = False
        for title in PERSON_TITLE_PREFIXES:
            if canonical.startswith(title) and len(canonical) > len(title) + 1:
                canonical = canonical[len(title):].strip(' ·•・._—-')
                changed = True
                break
        for title in PERSON_TITLE_SUFFIXES:
            if canonical.endswith(title) and len(canonical) > len(title) + 1:
                canonical = canonical[:-len(title)].strip(' ·•・._—-')
                changed = True
                break
    return canonical if canonical and not is_generic_person_name(canonical) else original

# ==================== Prompt 设计 ====================

EXTRACTOR_PROMPT = """你是一位严谨的文学档案员。请从给定小说片段提取真实出现的人物与关系，准确性优先，不虚构、不把地点、组织、物品当成人物。

规则：
1. 只提取有姓名、稳定专属称号或明确唯一身份的人物。不得把“共和国总统”“年轻军官”“绅士”“一位医生”“几个士兵”等数量词、修饰语或职业身份泛称当成人物。
   每个实体必须标注 entity_kind；国家、城市、地点、组织、家族、物品等必须使用非 person 类型，不能伪装成人物。
2. id 只使用不带职务/尊称的标准姓名；“奥雷里亚诺·布恩迪亚上校”“蒙卡达将军”“赫伯特先生”等完整称呼放入 aliases，并分别以“奥雷里亚诺·布恩迪亚”“蒙卡达”“赫伯特”作为 id。
3. 关系必须有片段内依据。每条关系最多保留 2 条 evidence，每条不超过 120 字；story 用不超过 120 字概括事件和关系变化。
4. 区分关系类型与强度。weight 表示关系紧密/剧情重要程度，不代表感情一定正面。
5. confidence 为 0 到 1；证据不充分时降低可信度，不能靠常识补写。
6. phase_id 和 phase_label 必须原样使用用户消息提供的剧情阶段。
7. 输出务必精炼：人物 description 不超过 80 字，不重复解释同一事实，不输出分析过程。
8. entities 中只保留至少参与一条有原文证据关系的人物；不要输出没有任何 relationships 的孤立实体。

输出 JSON 结构：
{
    "entities": [
        {"id": "标准名", "entity_kind": "person/location/organization/object/other", "type": "person", "aliases": ["别名"], "description": "身份/性格/关键行为", "phase_id": "剧情阶段ID", "phase_label": "剧情阶段名称"}
    ],
    "relationships": [
        {
            "source": "人物A",
            "target": "人物B",
            "relation": "具体关系描述",
            "type": "family/social/romance/conflict/work/other",
            "weight": 1到10的整数,
            "confidence": 0到1的小数,
            "story": "两人之间发生了什么以及关系如何变化",
            "evidence": ["原文证据1", "原文证据2"],
            "phase_id": "剧情阶段ID",
            "phase_label": "剧情阶段名称"
        }
    ]
}

type 字段说明：family=亲属, social=社交, romance=情感, conflict=冲突, work=工作/同僚, other=其他
weight 字段：关系紧密程度，1=边缘互动，10=贯穿主线的核心关系。"""

AGGREGATOR_PROMPT = """你是一位追求完美的数据归纳专家。你的任务是将碎片化的关系数据拼凑成一张**完整且无损**的宏大网络。

**最高指令：保留一切细节，严禁主观剔除！**

工作流程：
1. **实体对齐**：谨慎合并同一人物（如"Harry"和"Harry Potter"）。如果不确定是否为同一人，保留两个独立实体。合并后整合所有描述与 phases。
2. **关系叠加**：同一对人物同一类型的关系可以合并；不同关系类型分别保留。权重取综合强度（上限10），不要因为重复片段而简单相加。合并 story、evidence、phases，并把各阶段细节保存在 events 数组。
3. **网络完整性**：只保留真实人物及其有证据的关系。删除地点、国家、组织、物品等非人物实体，并删除没有参与任何关系的孤立实体。
4. **角色定性**：更新 type 字段：protagonist(主角), antagonist(反派), supporting(重要配角), neutral(普通角色)。

5. **证据约束**：保留输入中的直接证据，不新增输入里没有的事实。confidence 取证据链的合理综合值。

events 中每项使用 {"phase_id", "phase_label", "relation", "story", "evidence", "weight", "confidence"}。
请输出标准 JSON，包含完整的 "entities" 和 "relationships" 列表；entities 必须保留 entity_kind 字段。"""


class RelationshipExtractor:
    def __init__(self, llm_client=None):
        self.llm = llm_client or LLMClient()
        self.chunk_size = int(os.getenv('CHUNK_SIZE', '4500'))
        self.chunk_overlap = int(os.getenv('CHUNK_OVERLAP', '800'))
        self.max_workers = max(1, int(os.getenv('EXTRACT_MAX_WORKERS', '6')))
        self.max_retries = max(0, int(os.getenv('EXTRACT_MAX_RETRIES', '2')))
        self.max_phases = max(1, min(8, int(os.getenv('MAX_STORY_PHASES', '6'))))
        self.extract_max_tokens = max(1024, min(8192, int(os.getenv('EXTRACT_MAX_TOKENS', '4096'))))
        self.aggregation_mode = os.getenv('AGGREGATION_MODE', 'fast').strip().lower()
        self._last_extract_stats = {}

    def analyze(self, text, on_progress=None):
        """
        完整分析流程：预处理 → 分块 → 并行提取 → 递归聚合 → 后处理
        on_progress: 回调函数 (percent, message)
        返回: {nodes, links, overview}
        """
        # 1. 预处理 + 分块
        text = preprocess_text(text)
        chunks, phases = chunk_text_with_metadata(
            text, self.chunk_size, self.chunk_overlap, self.max_phases
        )
        total = len(chunks)

        if on_progress:
            on_progress(2, f'文本分块完成: {total} 块')

        # 2. 并行提取
        results = self._parallel_extract(chunks, on_progress)

        if on_progress:
            on_progress(90, '正在聚合合并...')

        # 3. 聚合。大文本默认使用纯代码聚合，避免额外的串行模型请求。
        if self.aggregation_mode == 'smart':
            merged = self._recursive_aggregate(results)
        else:
            merged = self._fast_merge(results)
        merged = self._fast_merge([merged])

        if on_progress:
            on_progress(96, '正在后处理...')

        # 4. 后处理 + 格式转换
        output = self._post_process(merged, phases=phases, source_length=len(text))
        output['overview']['analysis'] = dict(self._last_extract_stats)
        failed_chunks = self._last_extract_stats.get('failed_chunks', 0)
        if failed_chunks:
            output['overview']['warnings'] = [
                f'{failed_chunks} 个文本块在重试后仍提取失败，结果可能不完整。'
            ]

        if on_progress:
            on_progress(100, '分析完成')

        return output

    def _parallel_extract(self, chunks, on_progress=None):
        """并行提取所有文本块"""
        total = len(chunks)
        results_by_index = {}
        completed = 0
        last_error = None  # 记录最后一次错误，用于全部失败时抛出
        started_at = time.monotonic()

        # 动态调整并发数
        workers = min(self.max_workers, total)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {}
            for i, chunk in enumerate(chunks):
                future = executor.submit(self._extract_chunk_with_retry, i, chunk, total)
                futures[future] = i

            for future in as_completed(futures):
                chunk_index = futures[future]
                try:
                    result = future.result()
                    if result:
                        results_by_index[chunk_index] = result
                except Exception as e:
                    last_error = e
                    logger.error(f"块 {chunk_index} 提取失败: {e}")

                completed += 1
                if on_progress:
                    pct = min(int((completed / total) * 88) + 2, 89)
                    elapsed = max(time.monotonic() - started_at, 0.1)
                    remaining = max(0, round((elapsed / completed) * (total - completed)))
                    if remaining >= 60:
                        eta = f'约剩 {remaining // 60} 分 {remaining % 60:02d} 秒'
                    else:
                        eta = f'约剩 {remaining} 秒'
                    on_progress(pct, f'已提取 {completed}/{total} 块 · {eta}')

        # as_completed 按请求完成速度返回；聚合前必须恢复原文块顺序。
        results = [results_by_index[index] for index in sorted(results_by_index)]
        logger.info(f"并行提取完成: {len(results)}/{total} 块成功")
        self._last_extract_stats = {
            'total_chunks': total,
            'successful_chunks': len(results),
            'failed_chunks': total - len(results),
        }

        # 全部失败时抛出异常，避免返回空结果误导用户
        if not results:
            if last_error:
                # 提取关键错误信息（如 401 认证失败、429 限流等）
                err_msg = str(last_error)
                if '401' in err_msg or 'Authentication' in err_msg or 'invalid' in err_msg.lower():
                    raise RuntimeError('LLM API Key 无效或认证失败，请检查 .env 中的 LLM_API_KEY')
                elif '429' in err_msg or 'rate' in err_msg.lower():
                    raise RuntimeError('LLM 调用频率超限，请稍后重试或降低 EXTRACT_MAX_WORKERS')
                elif 'timeout' in err_msg.lower() or 'timed out' in err_msg.lower():
                    raise RuntimeError('LLM 调用超时，请检查网络或增大 request_timeout')
                else:
                    raise RuntimeError(f'所有文本块提取失败：{err_msg[:200]}')
            else:
                raise RuntimeError('所有文本块提取失败，但未捕获到具体错误')

        return results

    def _extract_chunk_with_retry(self, index, chunk, total):
        """对临时限流/网络错误进行有限退避重试。"""
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                return self._extract_chunk(index, chunk, total)
            except Exception as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    raise
                delay = min(8, 1.5 * (2 ** attempt))
                logger.warning(f'块 {index + 1} 提取失败，{delay:.1f} 秒后重试: {exc}')
                time.sleep(delay)
        raise last_error

    def _extract_chunk(self, index, chunk, total):
        """提取单个文本块的人物关系"""
        chunk_text = chunk['text']
        messages = [
            {"role": "system", "content": EXTRACTOR_PROMPT},
            {"role": "user", "content": (
                f"请分析以下文本片段（片段 {index+1}/{total}）。\n"
                f"剧情阶段：phase_id={chunk['phase_id']}，phase_label={chunk['phase_label']}\n"
                f"当前章节：{chunk['section']}\n\n{chunk_text}"
            )}
        ]
        result = self.llm.chat_json(
            messages, temperature=0.1, use_boost=True, max_tokens=self.extract_max_tokens
        )
        return self._normalize_result(
            result,
            default_phase_id=chunk['phase_id'],
            default_phase_label=chunk['phase_label'],
        )

    def _normalize_result(self, raw, default_phase_id='', default_phase_label=''):
        """标准化 LLM 返回的原始数据"""
        if not isinstance(raw, dict):
            return {'entities': [], 'relationships': []}
        entities = []
        for e in raw.get('entities', []):
            if not isinstance(e, dict):
                continue
            original_id = str(e.get('id', '')).strip()
            eid = normalize_person_name(original_id)
            if not eid or is_generic_person_name(original_id):
                continue
            aliases = e.get('aliases', [])
            if isinstance(aliases, str):
                aliases = re.split(r'[,，、;/；]+', aliases)
            aliases = list(aliases) if isinstance(aliases, list) else []
            if original_id != eid:
                aliases.append(original_id)
            entity_phases = []
            if isinstance(e.get('phases'), list):
                for phase in e['phases']:
                    if isinstance(phase, dict) and phase.get('id'):
                        entity_phases.append({
                            'id': str(phase['id']),
                            'label': str(phase.get('label') or phase['id']),
                        })
            if not entity_phases:
                entity_phases = [{
                    'id': str(e.get('phase_id') or default_phase_id or 'phase-1'),
                    'label': str(e.get('phase_label') or default_phase_label or '全文'),
                }]
            entities.append({
                'id': eid,
                'entity_kind': str(e.get('entity_kind', 'person')).strip().lower(),
                'type': e.get('type', 'person'),
                'aliases': self._unique_texts([
                    str(alias).strip() for alias in aliases
                    if str(alias).strip() and str(alias).strip() != eid
                    and not is_generic_person_name(alias)
                ], limit=30, max_length=60),
                'description': str(e.get('description', '')).strip(),
                'phases': entity_phases,
            })

        # 关系端点只能引用本批次已经确认的人物。这样即使模型同时输出了
        # “法国”等地点关系，也不会在后续聚合时被补成一个伪人物节点。
        person_alias_to_id = {}
        entities = [
            entity for entity in entities
            if entity.get('entity_kind', 'person') in {'person', 'human', 'character'}
        ]
        for entity in entities:
            person_alias_to_id[entity['id']] = entity['id']
            for alias in entity.get('aliases', []):
                person_alias_to_id.setdefault(alias, entity['id'])

        relationships = []
        for r in raw.get('relationships', []):
            if not isinstance(r, dict):
                continue
            src_raw = str(r.get('source', '')).strip()
            tgt_raw = str(r.get('target', '')).strip()
            src = person_alias_to_id.get(src_raw) or person_alias_to_id.get(normalize_person_name(src_raw))
            tgt = person_alias_to_id.get(tgt_raw) or person_alias_to_id.get(normalize_person_name(tgt_raw))
            if (not src or not tgt or src == tgt
                    or is_generic_person_name(src_raw) or is_generic_person_name(tgt_raw)):
                continue
            evidence = r.get('evidence', [])
            if isinstance(evidence, str):
                evidence = [item.strip() for item in re.split(r'\s*\|\s*', evidence) if item.strip()]
            elif not isinstance(evidence, list):
                evidence = []
            evidence = self._unique_texts(evidence, limit=8, max_length=240)
            if not evidence:
                continue
            weight = r.get('weight', 5)
            try:
                weight = max(1, min(10, int(weight)))
            except (ValueError, TypeError):
                weight = 5
            try:
                confidence = max(0.0, min(1.0, float(r.get('confidence', 0.7))))
            except (ValueError, TypeError):
                confidence = 0.7
            raw_phases = r.get('phases', [])
            if not isinstance(raw_phases, list):
                raw_phases = []
            phases = []
            for phase in raw_phases:
                if isinstance(phase, dict) and phase.get('id'):
                    phases.append({
                        'id': str(phase['id']),
                        'label': str(phase.get('label') or phase['id']),
                    })
                elif isinstance(phase, str) and phase.strip():
                    phases.append({'id': phase.strip(), 'label': phase.strip()})
            if not phases:
                phases = [{
                    'id': str(r.get('phase_id') or default_phase_id or 'phase-1'),
                    'label': str(r.get('phase_label') or default_phase_label or '全文'),
                }]
            events = []
            raw_events = r.get('events', [])
            if isinstance(raw_events, list):
                for event in raw_events:
                    if not isinstance(event, dict):
                        continue
                    event_evidence = event.get('evidence', [])
                    if isinstance(event_evidence, str):
                        event_evidence = [event_evidence]
                    events.append({
                        'phase_id': str(event.get('phase_id') or phases[0]['id']),
                        'phase_label': str(event.get('phase_label') or phases[0]['label']),
                        'relation': str(event.get('relation') or r.get('relation', '')).strip(),
                        'story': str(event.get('story') or '').strip(),
                        'evidence': self._unique_texts(event_evidence, limit=5, max_length=240),
                        'weight': weight,
                        'confidence': round(confidence, 2),
                    })
            if not events:
                for phase in phases:
                    events.append({
                        'phase_id': phase['id'],
                        'phase_label': phase['label'],
                        'relation': str(r.get('relation', '')).strip(),
                        'story': str(r.get('story', '')).strip(),
                        'evidence': evidence,
                        'weight': weight,
                        'confidence': round(confidence, 2),
                    })

            relationships.append({
                'source': src,
                'target': tgt,
                'relation': str(r.get('relation', '')).strip() or str(r.get('type', 'other')),
                'type': r.get('type', 'other'),
                'weight': weight,
                'confidence': round(confidence, 2),
                'story': str(r.get('story', '')).strip(),
                'evidence': evidence,
                'phases': phases,
                'events': events,
            })

        return {'entities': entities, 'relationships': relationships}

    @staticmethod
    def _unique_texts(values, limit=12, max_length=500):
        result = []
        seen = set()
        for value in values or []:
            text = re.sub(r'\s+', ' ', str(value)).strip()
            if not text:
                continue
            text = text[:max_length]
            key = text.casefold()
            if key in seen:
                continue
            seen.add(key)
            result.append(text)
            if len(result) >= limit:
                break
        return result

    # ==================== 递归聚合 ====================

    def _recursive_aggregate(self, results):
        """递归聚合：快速合并 → LLM 聚合"""
        if len(results) <= 1:
            return results[0] if results else {'entities': [], 'relationships': []}

        # 计算总字符数
        total_chars = sum(len(json.dumps(r, ensure_ascii=False)) for r in results)

        # 大量结果：先用快速合并减量
        if len(results) > 20 or total_chars > 50000:
            batch_size = 8 if len(results) > 20 else 4
            merged_batches = []
            for i in range(0, len(results), batch_size):
                batch = results[i:i + batch_size]
                merged_batches.append(self._fast_merge(batch))
            return self._recursive_aggregate(merged_batches)

        # 少量结果：直接 LLM 聚合
        if len(results) <= 4 and total_chars < 40000:
            return self._llm_aggregate(results)

        # 中等量：分批 LLM 聚合后递归
        batch_size = 4
        merged_batches = []
        for i in range(0, len(results), batch_size):
            batch = results[i:i + batch_size]
            if len(batch) == 1:
                merged_batches.append(batch[0])
            else:
                merged_batches.append(self._llm_aggregate(batch))
        return self._recursive_aggregate(merged_batches)

    def _fast_merge(self, results):
        """纯代码合并：用别名集合对齐人物，同时保留关系证据和剧情阶段。"""
        entity_records = [
            entity
            for result in results
            for entity in result.get('entities', [])
            if entity.get('id') and not is_generic_person_name(entity.get('id'))
        ]
        parent = {}

        def name_key(value):
            return re.sub(r'[\s·•・._—-]+', '', str(value)).casefold().strip('“”"\'()（）')

        def find(value):
            parent.setdefault(value, value)
            if parent[value] != value:
                parent[value] = find(parent[value])
            return parent[value]

        def union(left, right):
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                parent[right_root] = left_root

        id_frequency = {}
        original_forms = {}
        for entity in entity_records:
            entity_id = str(entity['id']).strip()
            entity_key = name_key(entity_id)
            if not entity_key:
                continue
            id_frequency[entity_id] = id_frequency.get(entity_id, 0) + 1
            original_forms.setdefault(entity_key, set()).add(entity_id)
            find(entity_key)
            for alias in entity.get('aliases', []):
                alias = str(alias).strip()
                alias_key = name_key(alias)
                if len(alias_key) < 2:
                    continue
                original_forms.setdefault(alias_key, set()).add(alias)
                union(entity_key, alias_key)

        groups = {}
        for entity in entity_records:
            key = name_key(entity['id'])
            if key:
                groups.setdefault(find(key), []).append(entity)

        entity_map = {}
        alias_index = {}
        role_priority = {'protagonist': 5, 'antagonist': 4, 'supporting': 3, 'neutral': 2, 'person': 1}
        for group_entities in groups.values():
            candidate_ids = [str(entity['id']).strip() for entity in group_entities]
            canonical = max(
                candidate_ids,
                key=lambda value: (id_frequency.get(value, 0), len(name_key(value))),
            )
            descriptions = self._unique_texts(
                [entity.get('description', '') for entity in group_entities], limit=6, max_length=300
            )
            aliases = []
            for entity in group_entities:
                aliases.extend(entity.get('aliases', []))
                if entity['id'] != canonical:
                    aliases.append(entity['id'])
            aliases = [item for item in self._unique_texts(aliases, limit=30, max_length=60) if item != canonical]
            entity_phases = []
            known_phase_ids = set()
            for entity in group_entities:
                for phase in entity.get('phases', []):
                    if phase.get('id') and phase['id'] not in known_phase_ids:
                        entity_phases.append(phase)
                        known_phase_ids.add(phase['id'])
            role = max(
                (entity.get('type', 'person') for entity in group_entities),
                key=lambda value: role_priority.get(value, 0),
            )
            entity_map[canonical] = {
                'id': canonical,
                'entity_kind': 'person',
                'type': role,
                'aliases': aliases,
                'description': ' '.join(descriptions),
                'phases': entity_phases,
            }
            for value in [canonical, *aliases, *candidate_ids]:
                alias_index[name_key(value)] = canonical

        relationship_map = {}
        for result in results:
            for relationship in result.get('relationships', []):
                source_raw = str(relationship.get('source', '')).strip()
                target_raw = str(relationship.get('target', '')).strip()
                if is_generic_person_name(source_raw) or is_generic_person_name(target_raw):
                    continue
                source = alias_index.get(name_key(source_raw), normalize_person_name(source_raw) or source_raw)
                target = alias_index.get(name_key(target_raw), normalize_person_name(target_raw) or target_raw)
                if not source or not target or source == target:
                    continue
                # 不再根据一条关系凭空补建端点；端点必须来自已经确认的人物实体。
                if source not in entity_map or target not in entity_map:
                    continue

                relation_type = str(relationship.get('type', 'other'))
                key = tuple(sorted((source, target))) + (relation_type,)
                existing = relationship_map.get(key)
                if not existing:
                    existing = {
                        'source': source,
                        'target': target,
                        'relation': '',
                        'type': relation_type,
                        'weight': 1,
                        'confidence': 0.0,
                        'story': '',
                        'evidence': [],
                        'phases': [],
                        'events': [],
                    }
                    relationship_map[key] = existing

                relation_names = self._unique_texts(
                    [existing['relation'], relationship.get('relation', '')], limit=4, max_length=80
                )
                existing['relation'] = '、'.join(relation_names)
                existing['weight'] = max(existing['weight'], int(relationship.get('weight', 5)))
                existing['confidence'] = max(existing['confidence'], float(relationship.get('confidence', 0.7)))
                stories = self._unique_texts(
                    [existing['story'], relationship.get('story', '')], limit=4, max_length=500
                )
                existing['story'] = '；'.join(stories)
                existing['evidence'] = self._unique_texts(
                    [*existing['evidence'], *relationship.get('evidence', [])], limit=12, max_length=240
                )

                phase_keys = {phase.get('id') for phase in existing['phases']}
                for phase in relationship.get('phases', []):
                    if phase.get('id') not in phase_keys:
                        existing['phases'].append(phase)
                        phase_keys.add(phase.get('id'))
                for event in relationship.get('events', []):
                    event_key = (
                        event.get('phase_id'), event.get('relation'),
                        ' '.join(event.get('evidence', []))
                    )
                    known_events = {
                        (item.get('phase_id'), item.get('relation'), ' '.join(item.get('evidence', [])))
                        for item in existing['events']
                    }
                    if event_key not in known_events:
                        existing['events'].append(event)

        return {
            'entities': list(entity_map.values()),
            'relationships': list(relationship_map.values()),
        }

    def _llm_aggregate(self, results):
        """LLM 智能聚合"""
        combined = json.dumps(results, ensure_ascii=False, indent=2)

        # 超限则回退到快速合并
        if len(combined) > 45000:
            logger.info("聚合输入超限，回退到快速合并")
            return self._fast_merge(results)

        messages = [
            {"role": "system", "content": AGGREGATOR_PROMPT},
            {"role": "user", "content": f"请合并以下提取结果：\n{combined}"}
        ]

        try:
            result = self.llm.chat_json(
                messages, temperature=0.1, use_boost=True,
                max_tokens=8192, request_timeout=120.0
            )
            return self._normalize_result(result)
        except Exception as e:
            logger.warning(f"LLM 聚合失败，回退到快速合并: {e}")
            return self._fast_merge(results)

    # ==================== 后处理 ====================

    def _post_process(self, data, phases=None, source_length=0):
        """
        后处理：角色推断、阵营检测、格式转换为前端兼容格式
        输出: {nodes, links, overview}
        """
        entities = data.get('entities', [])
        relationships = data.get('relationships', [])
        entities = [
            entity for entity in entities
            if entity.get('id')
            and entity.get('entity_kind', 'person') in {'person', 'human', 'character'}
            and not is_generic_person_name(entity.get('id'))
        ]
        relationships = [
            relationship for relationship in relationships
            if not is_generic_person_name(relationship.get('source'))
            and not is_generic_person_name(relationship.get('target'))
        ]
        phases = phases or [{'id': 'phase-1', 'label': '全文', 'order': 1}]
        phase_by_id = {phase['id']: phase for phase in phases}

        # 把别名端点归一化到规范人物 ID，并过滤真正不存在的节点。
        alias_to_id = {e['id']: e['id'] for e in entities}
        for e in entities:
            for alias in e.get('aliases', []):
                alias_to_id.setdefault(alias, e['id'])

        normalized_relationships = []
        for relationship in relationships:
            source = alias_to_id.get(relationship.get('source'))
            target = alias_to_id.get(relationship.get('target'))
            if source and target and source != target:
                normalized = dict(relationship)
                normalized['source'] = source
                normalized['target'] = target
                try:
                    normalized['weight'] = max(1, min(10, int(normalized.get('weight', 5))))
                except (ValueError, TypeError):
                    normalized['weight'] = 5
                normalized['evidence'] = self._unique_texts(
                    normalized.get('evidence', []), limit=12, max_length=240
                )
                normalized['story'] = str(normalized.get('story', '')).strip()
                try:
                    normalized['confidence'] = round(
                        max(0.0, min(1.0, float(normalized.get('confidence', 0.7)))), 2
                    )
                except (TypeError, ValueError):
                    normalized['confidence'] = 0.7
                normalized['phases'] = [
                    phase for phase in normalized.get('phases', [])
                    if isinstance(phase, dict) and phase.get('id') in phase_by_id
                ] or [phases[0]]
                normalized['events'] = [
                    event for event in normalized.get('events', [])
                    if isinstance(event, dict) and event.get('phase_id') in phase_by_id
                ]
                normalized_relationships.append(normalized)

        if len(normalized_relationships) < len(relationships):
            logger.warning(f"过滤了 {len(relationships) - len(normalized_relationships)} 条涉及不存在节点的关系")
        relationships = normalized_relationships

        # 关系图不展示零度实体。它们通常是地点误判、仅被顺带提及的人名，
        # 或模型未能给出证据关系的残留项；保留它们会被力布局甩到主网之外。
        connected_ids = {
            endpoint
            for relationship in relationships
            for endpoint in (relationship['source'], relationship['target'])
        }
        isolated_entities = [entity['id'] for entity in entities if entity['id'] not in connected_ids]
        if isolated_entities:
            logger.info(
                "过滤了 %s 个无有效关系的孤立实体: %s",
                len(isolated_entities), '、'.join(isolated_entities[:20])
            )
        entities = [entity for entity in entities if entity['id'] in connected_ids]

        # 角色推断（如果 LLM 未标记）
        self._infer_roles(entities, relationships)

        # 计算每个节点的度
        degree_map = {e['id']: 0 for e in entities}
        for r in relationships:
            degree_map[r['source']] = degree_map.get(r['source'], 0) + 1
            degree_map[r['target']] = degree_map.get(r['target'], 0) + 1

        entity_phases = {
            entity['id']: {
                phase.get('id') for phase in entity.get('phases', [])
                if isinstance(phase, dict) and phase.get('id') in phase_by_id
            }
            for entity in entities
        }
        for relationship in relationships:
            relation_phases = {phase['id'] for phase in relationship.get('phases', [])}
            entity_phases.setdefault(relationship['source'], set()).update(relation_phases)
            entity_phases.setdefault(relationship['target'], set()).update(relation_phases)

        # 转换为前端格式
        # 类别映射: protagonist=0, supporting=1, antagonist=2, family=3, neutral=4, other=5
        type_to_cat = {
            'protagonist': 0,
            'supporting': 1,
            'antagonist': 2,
            'family': 3,
            'neutral': 4,
            'person': 4,
            'other': 5
        }

        nodes = []
        for e in entities:
            cat = type_to_cat.get(e.get('type', 'person'), 4)
            desc = e.get('description', '')
            if e.get('aliases'):
                desc = f"别名: {'、'.join(e['aliases'])}。{desc}"
            nodes.append({
                'id': e['id'],
                'name': e['id'],
                'kind': 'person',
                'cat': cat,
                'era': 'all',
                'eras': sorted(entity_phases.get(e['id']) or {phase['id'] for phase in phases}),
                'aliases': list(e.get('aliases', [])),
                'desc': desc
            })

        links = []
        for r in relationships:
            links.append({
                's': r['source'],
                't': r['target'],
                'r': r.get('relation', r.get('type', 'other')),
                'w': r.get('weight', 5),
                'type': r.get('type', 'other'),
                'confidence': r.get('confidence', 0.7),
                'story': r.get('story', ''),
                'evidence': r.get('evidence', []),
                'phases': [phase['id'] for phase in r.get('phases', [])],
                'events': r.get('events', []),
            })

        # 生成概览
        overview = self._build_overview(entities, relationships, degree_map)

        return {
            'nodes': nodes,
            'links': links,
            'phases': phases,
            'overview': {
                **overview,
                'source_length': source_length,
                'phases': phases,
            },
            'schema_version': 3,
        }

    def _infer_roles(self, entities, relationships):
        """基于图论中心度推断主角/反派"""
        has_protagonist = any(e.get('type') == 'protagonist' for e in entities)
        has_antagonist = any(e.get('type') == 'antagonist' for e in entities)

        degree_map = {e['id']: 0 for e in entities}
        for r in relationships:
            degree_map[r['source']] = degree_map.get(r['source'], 0) + 1
            degree_map[r['target']] = degree_map.get(r['target'], 0) + 1

        if not has_protagonist and entities:
            protagonist = max(entities, key=lambda e: degree_map.get(e['id'], 0))
            protagonist['type'] = 'protagonist'
            logger.info(f"推断主角: {protagonist['id']}")

        if not has_antagonist and entities:
            # 找与主角有 conflict 关系的节点
            conflict_candidates = []
            protagonist_id = None
            for e in entities:
                if e.get('type') == 'protagonist':
                    protagonist_id = e['id']
                    break

            if protagonist_id:
                for r in relationships:
                    if r.get('type') != 'conflict':
                        continue
                    if r['source'] == protagonist_id:
                        conflict_candidates.append(r['target'])
                    elif r['target'] == protagonist_id:
                        conflict_candidates.append(r['source'])

            if conflict_candidates:
                antagonist = max(conflict_candidates, key=lambda x: degree_map.get(x, 0))
                for e in entities:
                    if e['id'] == antagonist:
                        e['type'] = 'antagonist'
                        logger.info(f"推断反派: {antagonist}")
                        break

    def _build_overview(self, entities, relationships, degree_map):
        """生成分析概览"""
        # 关系类型分布
        type_labels = {
            'family': '亲属', 'social': '社交', 'romance': '情感',
            'conflict': '冲突', 'work': '同僚', 'other': '其他'
        }
        type_dist = {}
        for r in relationships:
            label = type_labels.get(r.get('type', 'other'), '其他')
            type_dist[label] = type_dist.get(label, 0) + 1

        # 关键关系（按权重排序）
        key_rels = sorted(relationships, key=lambda x: x.get('weight', 0), reverse=True)[:10]

        # 主角关联
        protagonist = None
        for e in entities:
            if e.get('type') == 'protagonist':
                protagonist = e['id']
                break

        protagonist_connections = []
        if protagonist:
            for r in relationships:
                if r['source'] == protagonist or r['target'] == protagonist:
                    other = r['target'] if r['source'] == protagonist else r['source']
                    protagonist_connections.append({
                        'name': other,
                        'relation': r.get('relation', ''),
                        'weight': r.get('weight', 0)
                    })
            protagonist_connections.sort(key=lambda x: x['weight'], reverse=True)
            protagonist_connections = protagonist_connections[:6]

        return {
            'total_entities': len(entities),
            'total_relationships': len(relationships),
            'type_distribution': type_dist,
            'key_relationships': [
                f"{r['source']} ↔ {r['target']} ({r.get('relation', '')})"
                for r in key_rels
            ],
            'protagonist': protagonist,
            'protagonist_connections': protagonist_connections
        }
