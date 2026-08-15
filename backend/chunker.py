"""
文本分块模块
按句子边界优先分割，支持重叠以保留跨块上下文
"""
import re
import logging
from bisect import bisect_right

logger = logging.getLogger('silverfish.chunker')

CHAPTER_PATTERN = re.compile(
    r'(?m)^\s*((?:第[零〇一二三四五六七八九十百千万两\d]+[章节回卷部篇幕]|'
    r'(?:chapter|part|book)\s+[\divxlcdm]+)[^\n]{0,40})\s*$',
    re.IGNORECASE,
)


def preprocess_text(text: str) -> str:
    """预处理：标准化换行、压缩空行、去行首尾空白"""
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'\n{3,}', '\n\n', text)
    lines = [line.strip() for line in text.split('\n')]
    return '\n'.join(lines).strip()


def chunk_text(text: str, chunk_size: int = 4500, overlap: int = 800) -> list:
    """
    将长文本分块。
    策略：短文本直接返回；长文本按句子边界优先分割，带重叠。
    """
    if chunk_size <= 0:
        raise ValueError('CHUNK_SIZE 必须是大于 0 的整数')
    if overlap < 0:
        raise ValueError('CHUNK_OVERLAP 不能小于 0')
    if overlap >= chunk_size:
        raise ValueError('CHUNK_OVERLAP 必须小于 CHUNK_SIZE')

    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        if end >= len(text):
            chunk = text[start:]
            if len(chunk.strip()) > 10:
                chunks.append(chunk)
            break

        # 在块的后 50% 区域找句子边界
        search_start = start + int(chunk_size * 0.5)
        segment = text[search_start:end]

        # 优先级：段落分隔 > 句号 > 感叹号 > 问号
        separators = ['\n\n', '。', '！', '？', '.\n', '!\n', '?\n', '. ', '! ', '? ']
        best_pos = -1

        for sep in separators:
            pos = segment.rfind(sep)
            if pos != -1:
                best_pos = pos + len(sep)
                break

        if best_pos != -1:
            end = search_start + best_pos
        # 否则直接在 chunk_size 处截断

        chunk = text[start:end]
        if len(chunk.strip()) > 10:
            chunks.append(chunk)

        # 下一块从 end - overlap 开始
        next_start = end - overlap
        # 句子边界可能令实际块长小于 overlap；此时放弃本块重叠以保证前进。
        start = end if next_start <= start else next_start

    logger.info(f"文本分块完成: {len(text)} 字 -> {len(chunks)} 块")
    return chunks


def build_story_phases(text: str, max_phases: int = 6) -> list:
    """根据章节标题或文本位置生成可用于筛选的剧情阶段。"""
    if not text:
        return []
    max_phases = max(1, int(max_phases))

    headings = [
        {'title': match.group(1).strip(), 'start': match.start()}
        for match in CHAPTER_PATTERN.finditer(text)
    ]
    phases = []

    if len(headings) >= 2:
        group_count = min(max_phases, len(headings))
        for group_index in range(group_count):
            first_index = group_index * len(headings) // group_count
            next_index = (group_index + 1) * len(headings) // group_count
            last_index = max(first_index, next_index - 1)
            start = headings[first_index]['start']
            end = headings[next_index]['start'] if next_index < len(headings) else len(text)
            first_title = headings[first_index]['title']
            last_title = headings[last_index]['title']
            range_label = first_title if first_index == last_index else f'{first_title}–{last_title}'
            label = f'阶段{group_index + 1} · {range_label}'
            phases.append({
                'id': f'phase-{group_index + 1}',
                'label': label[:44],
                'start': start,
                'end': end,
                'order': group_index + 1,
            })
    else:
        labels = ['开篇', '发展', '转折', '高潮', '结局']
        if len(text) >= 40000:
            desired_count = 5
        elif len(text) >= 20000:
            desired_count = 4
        elif len(text) >= 8000:
            desired_count = 3
        else:
            desired_count = 1
        phase_count = min(len(labels), max_phases, desired_count)
        if phase_count == 1:
            selected_labels = ['全文']
        elif phase_count == 5:
            selected_labels = labels
        else:
            selected_labels = labels[:phase_count - 1] + [labels[-1]]
        for index in range(phase_count):
            phases.append({
                'id': f'phase-{index + 1}',
                'label': selected_labels[index],
                'start': index * len(text) // phase_count,
                'end': (index + 1) * len(text) // phase_count,
                'order': index + 1,
            })

    return phases


def chunk_text_with_metadata(text: str, chunk_size: int = 4500, overlap: int = 800,
                             max_phases: int = 6) -> tuple:
    """分块并附带字符位置、章节标题和剧情阶段，保持旧 chunk_text API 不变。"""
    chunks = chunk_text(text, chunk_size, overlap)
    phases = build_story_phases(text, max_phases=max_phases)
    metadata = []
    search_from = 0
    heading_matches = [
        (match.start(), match.group(1).strip())
        for match in CHAPTER_PATTERN.finditer(text)
    ]
    heading_starts = [item[0] for item in heading_matches]

    for index, chunk in enumerate(chunks):
        # 重叠块可能在更早位置再次出现；从上次位置附近搜索可稳定定位。
        start = text.find(chunk, max(0, search_from - overlap - 32))
        if start < 0:
            start = min(search_from, len(text))
        end = min(len(text), start + len(chunk))
        midpoint = (start + end) // 2
        phase = next(
            (item for item in phases if item['start'] <= midpoint < item['end']),
            phases[-1] if phases else {'id': 'phase-1', 'label': '全文'},
        )
        heading_index = bisect_right(heading_starts, start) - 1
        metadata.append({
            'index': index,
            'text': chunk,
            'start': start,
            'end': end,
            'section': heading_matches[heading_index][1] if heading_index >= 0 else phase['label'],
            'phase_id': phase['id'],
            'phase_label': phase['label'],
        })
        search_from = end

    public_phases = [
        {'id': item['id'], 'label': item['label'], 'order': item['order']}
        for item in phases
    ]
    return metadata, public_phases
