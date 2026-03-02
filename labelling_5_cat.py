import streamlit as st
import pandas as pd
import re
import os
from io import BytesIO
import streamlit.components.v1 as components

# --- 配置和关键词列表 ---

# 1. 默认列名
LABEL_COLUMN = "Human Label"
MODEL_LABEL_COLUMN = "gpt-label"
MODEL_TEXT_COLUMN = "gpt-text"
TEXT_COLUMN_DEFAULT = "text"
COMPANY_COLUMN_DEFAULT = "company"

# 2. 关键词列表 (用户提供的列表)
USER_KEYWORD_LIST = [
    "African American",
    "BLM",
    "D.E.I.",
    "DEI",
    "EDI",
    "LGBT",
    "Womany",
    "Women",
    "belonging",
    "bisexual",
    "black life matter",
    "disability",
    "disable",
    "diverse",
    "diversity",
    "equal",
    "equality",
    "equally",
    "equity",
    "ethnic",
    "ethnicity",
    "fair",
    "fairly",
    "fairness",
    "gay",
    "impartial",
    "impartiality",
    "impartially",
    "inclusion",
    "inclusive",
    "inclusivity",
    "justice",
    "lesbian",
    "queer",
    "race",
    "religion",
    "respect",
    "sexual identity",
    "sexual orientation",
    "transsexual",
    "welcomed",
    "不偏不倚",
    "不偏倚",
    "中性",
    "中立",
    "人权",
    "人權",
    "信仰",
    "全齡",
    "全龄",
    "公平",
    "公正",
    "共融",
    "包容",
    "友善",
    "友好",
    "双性恋",
    "受欢迎",
    "受歡迎",
    "同婚",
    "同志",
    "同性",
    "听障",
    "多元",
    "多样",
    "多樣",
    "失能",
    "女性",
    "妇女",
    "婦女",
    "宗教",
    "尊重",
    "工作与生活平衡",
    "工作生活平衡",
    "工作與生活平衡",
    "平权",
    "平權",
    "平等",
    "年老",
    "年長",
    "年长",
    "年齡",
    "年龄",
    "异性",
    "异质性",
    "弹性工时",
    "彈性工時",
    "归属感",
    "心障者",
    "性倾向",
    "性傾向",
    "性別",
    "性别",
    "性取向",
    "性少数",
    "性少數",
    "性認同",
    "性认同",
    "性身份認同",
    "性身份认同",
    "性身分認同",
    "性身分认同",
    "接納",
    "接纳",
    "族群",
    "族裔",
    "无障碍",
    "正义",
    "正義",
    "歧視",
    "歧视",
    "歸屬感",
    "残疾",
    "残障",
    "殘疾",
    "殘障",
    "民族",
    "無障礙",
    "熟齡",
    "熟龄",
    "男女",
    "男性",
    "異性",
    "異質性",
    "种族",
    "種族",
    "穷人",
    "窮人",
    "老年",
    "聽障",
    "衡平",
    "視障",
    "视障",
    "評選",
    "評鑑",
    "認同感",
    "认同感",
    "评选",
    "评鉴",
    "跨性別",
    "跨性别",
    "跨文化",
    "身障",
    "酷儿",
    "酷兒",
    "障碍",
    "障礙",
    "雙性戀",
    "非洲裔美国人",
    "非洲裔美國人",
    "高齡",
    "高龄",
    "黑人的命也是命",
    "黑命关天",
    "黑命關天",
    "黑种人",
    "黑種人"
]

# 3. 公司后缀列表 (用于清理公司全名)
COMPANY_SUFFIXES = [
    r'\bINC\b', r'\bCORP\b', r'\bLTD\b', r'\bCO\b', r'\bLLC\b', 
    r'\bPLC\b', r'\bS\.A\.\b', r'\bGMBH\b', r'\bA\.G\.\b', 
    r'\bGROUP\b', r'\bCOMPANIES\b', r'\bCOMPANY\b', r'\bCP\b', r'\bCL\b', r'\bSE\b'
]
COMPANY_SUFFIX_REGEX = re.compile('|'.join(COMPANY_SUFFIXES), re.IGNORECASE)

# 4. Define the special tokens and their replacement HTML structures
KEYWORD_START = "<KEYWORD>"
KEYWORD_END = "</KEYWORD>"
COMPANY_START = "<COMPANY>"
COMPANY_END = "</COMPANY>"

KEYWORD_MARK_START = "<mark>"
KEYWORD_MARK_END = "</mark>"
COMPANY_MARK_START = '<span style="background-color: lightblue; padding: 2px 4px; border-radius: 3px;">'
COMPANY_MARK_END = '</span>'


# --- Navigation Functions ---

def go_to_index(new_idx):
    """跳转到指定索引，并重新运行应用。"""
    if 'df' not in st.session_state or st.session_state.df.empty:
        st.sidebar.error("请先加载数据。")
        return
        
    df_len = len(st.session_state.df)
    if 0 <= new_idx < df_len:
        st.session_state.current_index = new_idx
        st.rerun()
    elif new_idx >= df_len:
        st.sidebar.warning("已是最后一项。")
    elif new_idx < 0:
        st.sidebar.warning("已是第一项。")

def go_to_next():
    """跳转到下一项。"""
    go_to_index(st.session_state.current_index + 1)

def go_to_previous():
    """跳转到上一项。"""
    go_to_index(st.session_state.current_index - 1)

def go_to_next_unlabeled():
    """跳转到下一未标注项。"""
    if 'df' not in st.session_state or st.session_state.df.empty:
        st.sidebar.error("请先加载数据。")
        return
        
    current_idx = st.session_state.current_index
    unlabeled_df = st.session_state.df[st.session_state.df[LABEL_COLUMN] == -1]
    
    if unlabeled_df.empty:
        st.sidebar.success("所有数据已标注完毕！")
        return

    next_unlabeled_index = unlabeled_df[unlabeled_df.index > current_idx].index.min()

    if pd.isna(next_unlabeled_index):
        next_unlabeled_index = unlabeled_df.index.min()
        
    if pd.notna(next_unlabeled_index) and next_unlabeled_index != current_idx:
        go_to_index(next_unlabeled_index)
    elif pd.isna(next_unlabeled_index):
        st.sidebar.success("所有数据已标注完毕！")
    else:
        st.sidebar.warning("没有更多的未标注项了。")


def go_to_last_unlabeled():
    """跳转到上一未标注项。"""
    if 'df' not in st.session_state or st.session_state.df.empty:
        st.sidebar.error("请先加载数据。")
        return
        
    current_idx = st.session_state.current_index
    last_unlabeled_index = st.session_state.df[
        (st.session_state.df[LABEL_COLUMN] == -1) & (st.session_state.df.index < current_idx)
    ].index.max()

    if pd.notna(last_unlabeled_index):
        go_to_index(last_unlabeled_index)
    else:
        st.sidebar.warning("没有更早的未标注项了。")

# --- Data Loading and Saving Functions ---

@st.cache_data(show_spinner="正在读取文件...")
def load_data_file(uploaded_file):
    """从上传的文件读取原始数据，不进行任何处理。"""
    try:
        if uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            st.error("不支持的文件类型。请上传 .xlsx 或 .csv 文件。")
            return None
        return df
    except Exception as e:
        st.error(f"读取文件失败: {e}")
        return None

def initialize_session_state_df(df_raw, text_col, company_col):
    """
    处理原始DataFrame：
    1. 校验列名。
    2. 检查并生成 Human Label 列。
    3. 初始化 Session State 并跳转。
    """
    if text_col not in df_raw.columns or company_col not in df_raw.columns:
        st.error(f"文件中缺少必需的列。请检查您选择的 '{text_col}' 和 '{company_col}'。")
        return

    df = df_raw.copy()
    
    # 1. 检查并生成/清理 Human Label 列
    label_status = ""
    if LABEL_COLUMN not in df.columns:
        df[LABEL_COLUMN] = -1
        label_status = f"文件未包含 '{LABEL_COLUMN}' 列，已自动生成并开始全新标注。"
    else:
        # 清理现有标注列
        original_unlabeled_count = (df[LABEL_COLUMN].fillna(-1) == -1).sum()
        df[LABEL_COLUMN] = df[LABEL_COLUMN].fillna(-1).astype(int)
        new_unlabeled_count = (df[LABEL_COLUMN] == -1).sum()
        
        if original_unlabeled_count == new_unlabeled_count:
             label_status = f"文件包含 '{LABEL_COLUMN}' 列，继续上次标注 ({new_unlabeled_count} 项未标注)。"
        else:
             label_status = f"文件包含 '{LABEL_COLUMN}' 列，已将空值填充为 -1，继续上次标注 ({new_unlabeled_count} 项未标注)。"


    # 2. 设置 Session State
    st.session_state.df = df
    st.session_state.TEXT_COLUMN = text_col
    st.session_state.COMPANY_COLUMN = company_col
    
    if MODEL_TEXT_COLUMN in df.columns:
        st.session_state.MODEL_TEXT_COLUMN = MODEL_TEXT_COLUMN
        st.session_state.display_model_text = True
    else:
        st.session_state.display_model_text = False

    st.session_state.file_name = st.session_state.get('uploaded_file_name', '未命名')
    st.session_state.label_status = label_status

    # 3. 确定起始索引：跳转到第一个未标注项
    first_unlabeled = df[df[LABEL_COLUMN] == -1].index.min()
    if pd.notna(first_unlabeled):
        st.session_state.current_index = first_unlabeled
    elif not df.empty:
        st.session_state.current_index = 0 # 所有已标注，跳转到第一项
    else:
        st.session_state.current_index = -1 # 数据为空

    st.rerun()


def save_data(df, auto_save=True):
    """将 DataFrame 保存到内存中的 Excel 文件。"""
    try:
        # 使用 BytesIO 在内存中创建 Excel 文件
        output = BytesIO()
        df.to_excel(output, index=False)
        output.seek(0)
        st.session_state.saved_data = output
        
        if not auto_save:
            st.success(f"**人工保存成功！** 标注结果已保存到内存。请使用下方的下载按钮获取最新文件。")
    except Exception as e:
        st.error(f"**保存失败！** 错误: {e}")

# --- Utility Functions ---

def build_keyword_regex(keyword_list):
    """
    Builds case-sensitive (CS) and case-insensitive (CI) regex patterns,
    ensuring strict word boundaries for acronyms DEI, EDI, and BLM.
    """
    cs_keywords = []
    ci_patterns = []

    for keyword in keyword_list:
        keyword = keyword.strip()
        if not keyword:
            continue
            
        # 1. Case-Sensitive Acronyms (DEI, EDI, BLM)
        if keyword in ["DEI", "EDI", "BLM"]:
            # Use \b to ensure it's treated as a whole word (e.g., prevents M(EDI)A match)
            cs_keywords.append(r'\b' + re.escape(keyword) + r'\b')
        
        # 2. Case-Insensitive Keywords
        else:
            escaped_kw = re.escape(keyword)
            
            # If the original logic included wildcard support ('*'), use the replacement:
            escaped_kw = escaped_kw.replace(r'\*', r'\w*')
            
            if ' ' in escaped_kw or r'\w*' in escaped_kw:
                # Phrase match or Wildcard match: don't enforce leading/trailing word boundaries
                ci_patterns.append(escaped_kw)
            else:
                # Single word match: enforce word boundaries for accuracy
                ci_patterns.append(r'(?:(?<![a-zA-Z0-9_])' + escaped_kw + r'(?![a-zA-Z0-9_]))')

    cs_pattern = "|".join(cs_keywords)
    ci_pattern = "|".join(ci_patterns)
    
    return cs_pattern, ci_pattern

#def clean_company_name(company_name):
#    """Remove common company suffixes to get the base name."""
#    if pd.isna(company_name):
#        return ""
#    cleaned_name = COMPANY_SUFFIX_REGEX.sub('', str(company_name)).strip()
#    cleaned_name = re.sub(r'[^\w\s]', '', cleaned_name)
#    cleaned_name = re.sub(r'\s+', ' ', cleaned_name).strip()
#    
#    words = [re.escape(word) for word in cleaned_name.split()]
#    return "|".join(words)
    
def clean_company_name(company_name):
    """
    Builds a regex pattern for company name matching, including variants 
    with and without internal punctuation.
    """
    if pd.isna(company_name) or not company_name.strip():
        return ""
    base_name = str(company_name).strip()
    name_without_suffix = COMPANY_SUFFIX_REGEX.sub('', base_name).strip()
    raw_pattern = re.escape(name_without_suffix).replace(r'\ ', r'\s*')

    cleaned_words_str = re.sub(r'[^\w\s]', ' ', name_without_suffix)
    cleaned_words_str = re.sub(r'\s+', ' ', cleaned_words_str).strip()
    word_patterns = [re.escape(word) for word in cleaned_words_str.split() if (len(word) >= 2 and word not in ['AND', 'THE', 'FOR', 'WITH', 'DR', 'OF'])]
    
    all_patterns = set(word_patterns)
    if raw_pattern:
         all_patterns.add(raw_pattern)

    return "|".join(filter(None, all_patterns))

def highlight_text(text, company_name, cs_pattern, ci_pattern):
    """
    Highlights text using HTML marks.
    If special tokens are detected, they are replaced with HTML marks.
    Otherwise, the text is processed using the original regex highlighting logic.
    """
    if not text: 
        return ""

    text = str(text)

    # 1. Check for Special Tokens
    # If the text contains any special tokens, we assume it's pre-tokenized and switch mode.
    is_tokenized = KEYWORD_START in text or COMPANY_START in text

    if is_tokenized:
        # --- MODE 1: Replace Tokens with HTML Marks ---
        
        # 1. Company Replacement
        # Regex to find: [COMPANY_START]...[COMPANY_END] (non-greedy match for content)
        token_co_pattern = re.escape(COMPANY_START) + r'(.*?)' + re.escape(COMPANY_END)
        def token_co_replacer(match):
            # The content is in group(1)
            return f'{COMPANY_MARK_START}{match.group(1)}{COMPANY_MARK_END}'
        
        text = re.sub(token_co_pattern, token_co_replacer, text, flags=re.DOTALL) # DOTALL handles multiline if it were an issue

        # 2. Keyword Replacement
        # Regex to find: [KEYWORD_START]...[KEYWORD_END]
        token_kw_pattern = re.escape(KEYWORD_START) + r'(.*?)' + re.escape(KEYWORD_END)
        def token_kw_replacer(match):
            # The content is in group(1)
            return f'{KEYWORD_MARK_START}{match.group(1)}{KEYWORD_MARK_END}'
            
        text = re.sub(token_kw_pattern, token_kw_replacer, text, flags=re.DOTALL)
        
        # Finally, replace newlines for HTML display
        text = text.replace('\n', '<br>')
        return text

    else:
        # --- MODE 2: Original Regex Highlighting (Raw Text) ---
        
        text = text.replace('\n', '<br>')
        
        company_patterns = clean_company_name(company_name)
        
        # Step 1: Replace company matches with unique placeholders
        company_placeholders = {}
        placeholder_counter = [0]
        
        def company_replacer_mode2(match):
            placeholder = f'\x00COMPANY_{placeholder_counter[0]}\x00'
            company_placeholders[placeholder] = f'{COMPANY_MARK_START}{match.group(0)}{COMPANY_MARK_END}'
            placeholder_counter[0] += 1
            return placeholder

        if company_patterns:
            full_company_pattern = r'(?:(?<![a-zA-Z0-9_])(?:' + company_patterns + r')(?![a-zA-Z0-9_]))'
            text = re.sub(full_company_pattern, company_replacer_mode2, text, flags=re.IGNORECASE)

        # Step 2: Apply keyword highlighting (won't touch placeholders)
        def keyword_replacer_mode2(match):
            return f"{KEYWORD_MARK_START}{match.group(0)}{KEYWORD_MARK_END}"

        if cs_pattern:
            text = re.sub(f"({cs_pattern})", keyword_replacer_mode2, text)
        if ci_pattern:
            text = re.sub(f"({ci_pattern})", keyword_replacer_mode2, text, flags=re.IGNORECASE)
        
        # Step 3: Restore company placeholders with actual HTML
        for placeholder, html in company_placeholders.items():
            text = text.replace(placeholder, html)
            
        return text

# --- Chart Calculation and Drawing Function ---

def create_and_show_stacked_bar(df):
    """
    计算各标签的百分比，并使用 HTML/CSS 创建一个彩色的水平堆叠条形图，
    并将图例显示为垂直排列的表格。
    """
    total_count = len(df)
    if total_count == 0:
        st.write("数据框为空，无法显示统计信息。")
        return

    # 标签映射和颜色配置
    # 注意：我们按 0, 1, 2, 3, 4, -1 的顺序处理，以便堆叠和表格渲染
    label_info = {
        0: {'name': '0 (不相关)', 'color': '#FFFFFF'},
        1: {'name': '1 (继续)', 'color': '#28A745'},
        2: {'name': '2 (减少)', 'color': '#DC3545'},
        3: {'name': '3 (不明确)', 'color': '#FFC107'},
        4: {'name': '4（重新包装)', 'color': '#4682B4'}, # 修正颜色格式
        -1: {'name': '-1 (未标注)', 'color': '#AAAAAA'}
    }

    # 统计每个标签的数量并计算百分比
    counts = df[LABEL_COLUMN].value_counts().reindex(list(label_info.keys()), fill_value=0)
    percentages = (counts / total_count * 100).round(1)
    
    # 将计数和百分比信息添加到 label_info 中，方便后续表格渲染
    for label, info in label_info.items():
        info['count'] = counts.get(label, 0)
        info['percent'] = percentages.get(label, 0)

    # 1. 创建堆叠条形图的 HTML
    html_bar = '<div style="width: 100%; height: 25px; border-radius: 5px; overflow: hidden; display: flex; margin-bottom: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">'
    
    # 按照 label_info 的顺序遍历创建条形
    for label, info in label_info.items():
        percent = info['percent']
        if percent > 0:
            html_bar += f'<div style="background-color: {info["color"]}; width: {percent}%; height: 100%;" title="{info["name"]}: {info["count"]} ({percent:.1f}%)"></div>'
            
    html_bar += '</div>'
    
    # 2. 创建图例 HTML (表格形式)
    
    # 表格头部样式
    table_style = """
        <style>
            .legend-table {
                width: 100%; 
                border-collapse: collapse; 
                font-size: 14px; 
                margin-top: 10px;
            }
            .legend-table th, .legend-table td {
                padding: 6px 12px;
                text-align: left;
                border: 1px solid #e0e0e0;
            }
            .legend-table th {
                background-color: #f8f9fa;
                font-weight: bold;
            }
            .color-box {
                width: 12px; 
                height: 12px; 
                border-radius: 3px; 
                display: inline-block; 
                border: 1px solid #ccc;
            }
        </style>
    """

    # 表格内容
    table_rows = []
    # 表格标题行
    table_rows.append('<tr><th>颜色</th><th>标签</th><th>具体数值</th><th>百分比</th></tr>')

    # 遍历 label_info 创建表格行
    for label, info in label_info.items():
        # 排除 count 为 0 且 percent 为 0 的情况（如果需要，可以保留 0% 的行）
        if info['count'] > 0 or info['percent'] > 0: 
            table_rows.append(
                f'<tr>'
                f'<td><span class="color-box" style="background-color: {info["color"]};"></span></td>'
                f'<td>{info["name"]}</td>'
                f'<td>{info["count"]}</td>'
                f'<td>**{info["percent"]:.1f}%**</td>'
                f'</tr>'
            )

    html_legend = f'{table_style}<table class="legend-table">{"".join(table_rows)}</table>'

    st.subheader("标注进度概览")
    
    # 显示总进度
    labeled_count = total_count - counts.get(-1, 0)
    st.metric(
        "总进度", 
        f"{labeled_count} / {total_count}", 
        delta=f"{labeled_count / total_count * 100:.1f}% 已完成"
    )

    # 使用 Markdown/HTML 显示条形图和表格图例
    st.markdown(html_bar, unsafe_allow_html=True)
    st.markdown(html_legend, unsafe_allow_html=True)


# --- Streamlit Session State Initialization and Labeling Logic ---

# 检查高亮关键词
cs_pattern, ci_pattern = build_keyword_regex(USER_KEYWORD_LIST)


def handle_label_input(label):
    """处理用户输入，保存标签，并自动跳转到下一未标注项。"""
    idx = st.session_state.current_index
    if 0 <= idx < len(st.session_state.df):
        st.session_state.df.loc[idx, LABEL_COLUMN] = label
        save_data(st.session_state.df, auto_save=True)

        # 查找下一个未标注项 (优先查找当前项之后)
        next_unlabeled_index = st.session_state.df[
            (st.session_state.df[LABEL_COLUMN] == -1) & 
            (st.session_state.df.index > idx)
        ].index.min()

        # 如果当前之后没有，则从头查找最小的未标注索引
        if pd.isna(next_unlabeled_index):
             next_unlabeled_index = st.session_state.df[st.session_state.df[LABEL_COLUMN] == -1].index.min()

        if pd.isna(next_unlabeled_index):
            st.session_state.current_index = -1 # 所有标注完成
        else:
            st.session_state.current_index = next_unlabeled_index
        
        st.rerun() 

# --- Streamlit Interface ---

st.title("DEI标注工具 (4类别)")

# --- Sidebar for Data Loading and Column Selection ---
with st.sidebar:
    st.header("1. 数据加载与配置")
    
    uploaded_file = st.file_uploader(
        "上传包含标注数据的 Excel 或 CSV 文件",
        type=['xlsx', 'csv'],
        help=f"文件应包含文本内容列和公司名列。标注结果将写入名为 '{LABEL_COLUMN}' 的列中。"
    )
    
    # 状态：文件已上传
    if uploaded_file is not None:
        if 'uploaded_file_name' not in st.session_state or st.session_state.uploaded_file_name != uploaded_file.name:
            st.session_state.uploaded_file_name = uploaded_file.name
            st.session_state.df_raw = load_data_file(uploaded_file)
            if 'df' in st.session_state:
                del st.session_state.df
                
        df_raw = st.session_state.get('df_raw')

        if df_raw is not None and 'df' not in st.session_state:
            # 文件已上传且原始数据已加载，但主df未配置，进入列选择阶段
            st.markdown("---")
            st.subheader("选择数据列")

            all_cols = list(df_raw.columns)
            
            default_text = TEXT_COLUMN_DEFAULT if TEXT_COLUMN_DEFAULT in all_cols else (all_cols[0] if len(all_cols)>0 else None)
            default_company = COMPANY_COLUMN_DEFAULT if COMPANY_COLUMN_DEFAULT in all_cols else (all_cols[1] if len(all_cols)>1 else default_text)


            text_col_select = st.selectbox(
                "请选择**文本内容列**:",
                all_cols,
                index=all_cols.index(default_text) if default_text in all_cols else 0,
                key='text_col_select'
            )
            
            company_col_select = st.selectbox(
                "请选择**公司名列**:",
                all_cols,
                index=all_cols.index(default_company) if default_company in all_cols else (1 if len(all_cols) > 1 else 0),
                key='company_col_select'
            )

            if st.button("🚀 **加载数据并开始标注**", type="primary", use_container_width=True):
                if text_col_select == company_col_select:
                    st.error("文本内容列和公司名列不能相同，请重新选择。")
                else:
                    initialize_session_state_df(df_raw, text_col_select, company_col_select)
        
    
    # 状态：主DF已加载
    if 'df' in st.session_state:
        df = st.session_state.df
        total_count = len(df)
        current_idx = st.session_state.current_index

        # 1. Status and Save
        st.subheader("数据操作")
        
        # 保存/下载按钮
        col_save, col_download = st.columns(2)
        with col_save:
            if st.button("💾保存到内存", type="primary", use_container_width=True):
                save_data(df, auto_save=False)
                
        with col_download:
            if 'saved_data' in st.session_state:
                st.download_button(
                    label="⬇️下载标注结果",
                    data=st.session_state.saved_data,
                    file_name=f"labeled_{st.session_state.file_name.replace('.xlsx', '').replace('.csv', '')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            else:
                st.button("⬇️下载标注结果", disabled=True, use_container_width=True)

        st.markdown("---")
        
        st.header("2. 状态与操作")
        st.info(f"文件名: **{st.session_state.file_name}**")
        st.caption(st.session_state.get('label_status', ''))
        
        # Display Stacked Bar Chart
        if total_count > 0:
            create_and_show_stacked_bar(df)
        
        st.markdown("---")
        

        
        # 3. Labeling Buttons (4 categories)
        if current_idx != -1:
            st.subheader(f"标注 (ID: {current_idx + 1}/{total_count})")
            
            # Labeling UI 
            col1, col2 = st.columns(2)
            with col1:
                if st.button("1 (继续)", use_container_width=True, help="维持/继续DEI/EDI相关指标"):
                    handle_label_input(1)
            with col2:
                if st.button("2 (减少)", use_container_width=True, help="DEI/EDI相关指标出现减少"):
                    handle_label_input(2)
            
            col3, col4 = st.columns(2)
            with col3:
                if st.button("3 (不明确)", use_container_width=True, help="文本未明确提及变化"):
                    handle_label_input(3)
            with col4:
                if st.button("4 (重新包装)", use_container_width=True, help="重新包装DEI相关表述"):
                    handle_label_input(4)

            col5, _ = st.columns([1, 1])
            with col5:
                if st.button("0 (不相关)", use_container_width=True, help="文本内容与DEI/EDI主题不相关"):
                    handle_label_input(0)
                    
            label_map = {-1: '未标注', 1: '继续', 2: '减少', 3: '不明确', 4: '重新包装', 0: '不相关'}
            
            previous_label = df.loc[current_idx, LABEL_COLUMN]
            st.info(f"当前标签: **{label_map.get(previous_label, '未知')}**")

            if MODEL_LABEL_COLUMN in df.columns:
                model_label = df.loc[current_idx, MODEL_LABEL_COLUMN]      
                st.info(f"机器标签: **{label_map.get(model_label, '未知')}**")
            
        else:
            st.success("所有数据已标注完毕！")
            
        st.markdown("---")

        # 4. Navigation Controls
        st.subheader("3. 导航")
        
        col_prev, col_next = st.columns(2)
        with col_prev:
            if st.button("上一项", use_container_width=True, disabled=(current_idx <= 0 or current_idx == -1)):
                go_to_previous()
        with col_next:
            if st.button("下一项", use_container_width=True, disabled=(current_idx >= total_count - 1 or current_idx == -1)):
                go_to_next()
                
        col_last_un, col_next_un = st.columns(2)
        with col_last_un:
            if st.button("上一未标注", use_container_width=True, disabled=(current_idx == -1)):
                go_to_last_unlabeled()
        with col_next_un:
            if st.button("下一未标注", use_container_width=True, disabled=(current_idx == -1)):
                go_to_next_unlabeled()

        st.markdown("---")
        jump_id = st.number_input(
            "跳转到 ID (1-based)", 
            min_value=1, 
            max_value=total_count, 
            value=current_idx + 1 if current_idx != -1 else 1,
            step=1,
            key='jump_input'
        )
        if st.button(f"跳转到 #{jump_id}", use_container_width=True):
            go_to_index(jump_id - 1)

        st.markdown("---")
        st.subheader("4. 键盘快捷键")
        st.markdown("""
| 按键 | 功能 |
|------|------|
| `1` | 继续 |
| `2` | 减少 |
| `3` | 不明确 |
| `4` | 重新包装 |
| `0` | 不相关 |
| `→` / `n` | 下一项 |
| `←` / `p` | 上一项 |
| `u` | 下一未标注 |
""")
        st.markdown("---")
        st.subheader("5. 高亮规则")
        st.markdown(f"- **蓝色:** 公司名关键词 (`{st.session_state.COMPANY_COLUMN}` 提取)")
        st.markdown("- **黄色:** DEI/EDI 关键词")
        st.markdown("- **全大写**关键词区分大小写。")


# --- Main Content Display (unchanged) ---
if 'df' not in st.session_state:
    st.info("请在左侧上传文件并配置数据列，以开始标注。")
elif st.session_state.current_index == -1 and not st.session_state.df.empty:
    st.success("**恭喜！所有数据标注完成！**")
    st.dataframe(st.session_state.df)
else:
    df = st.session_state.df
    current_idx = st.session_state.current_index
    total_count = len(df)

    # 使用 Session State 中存储的动态列名
    COMPANY_COLUMN_ACTIVE = st.session_state.COMPANY_COLUMN
    TEXT_COLUMN_ACTIVE = st.session_state.TEXT_COLUMN
    
    current_company = df.loc[current_idx, COMPANY_COLUMN_ACTIVE]
    current_text = df.loc[current_idx, TEXT_COLUMN_ACTIVE]
    
    st.header(f"文本 ID: {current_idx + 1} / {total_count}")
    st.markdown(f"**文件:** `{st.session_state.file_name}`")
    st.markdown("---")

    # 1. Display Company Name
    st.markdown(f"#### 公司名称 (`{COMPANY_COLUMN_ACTIVE}`):")
    st.code(current_company, language="")

    if st.session_state.display_model_text:
        GPT_TEXT_COLUMN_ACTIVE = st.session_state.MODEL_TEXT_COLUMN
        current_model_text = df.loc[current_idx, GPT_TEXT_COLUMN_ACTIVE]
        st.markdown(f"#### 机器预测说明 (`{GPT_TEXT_COLUMN_ACTIVE}`):")
        st.markdown(
        f'<div style="background-color: #f0f2f6; padding: 15px; border-radius: 5px; border: 1px solid #e0e0e0; font-size: 110%; line-height: 1.6;">'
        f'{current_model_text}'
        f'</div>',
        unsafe_allow_html=True
        )
        

    # 2. Display Highlighted Extracted Text
    highlighted_text = highlight_text(current_text, current_company, cs_pattern, ci_pattern)
    st.markdown(f"#### 待标注文本 (`{TEXT_COLUMN_ACTIVE}`) - 关键词已高亮:")
    
    st.markdown(
        f'<div style="background-color: #f0f2f6; padding: 15px; border-radius: 5px; border: 1px solid #e0e0e0; font-size: 110%; line-height: 1.6;">'
        f'{highlighted_text}'
        f'</div>',
        unsafe_allow_html=True
    )

    # --- Keyboard Shortcuts ---
    keyboard_js = """
    <script>
    (function() {
        const KEY_MAP = {
            '0': '0 (不相关)',
            '1': '1 (继续)',
            '2': '2 (减少)',
            '3': '3 (不明确)',
            '4': '4 (重新包装)',
            'n': '下一项',
            'p': '上一项',
            'u': '下一未标注',
            'ArrowRight': '下一项',
            'ArrowLeft': '上一项'
        };

        function findAndClickButton(targetText) {
            const doc = window.parent.document;
            const buttons = doc.querySelectorAll('button');
            for (const btn of buttons) {
                if (btn.innerText.trim() === targetText) {
                    if (btn.disabled) return false;
                    btn.click();
                    return true;
                }
            }
            return false;
        }

        const parentDoc = window.parent.document;

        // Remove previous listener if it exists
        if (parentDoc._labelKeyHandler) {
            parentDoc.removeEventListener('keydown', parentDoc._labelKeyHandler);
        }

        parentDoc._labelKeyHandler = function(event) {
            if (event.ctrlKey || event.altKey || event.metaKey) return;
            const tag = event.target.tagName;
            if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;

            const key = event.key;
            const targetText = KEY_MAP[key] || KEY_MAP[key.toLowerCase()];
            if (targetText) {
                event.preventDefault();
                findAndClickButton(targetText);
            }
        };

        parentDoc.addEventListener('keydown', parentDoc._labelKeyHandler);
    })();
    </script>
    """
    components.html(keyboard_js, height=0)
