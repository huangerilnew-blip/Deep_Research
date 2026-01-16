import pandas as pd
import os
import json
from pathlib import Path
from config import Config


def extract_company_info(base_path: str = None, columns_map: dict[str, str] = None,save_path: str = None) :
    """
    从指定路径下的所有CSV文件中提取企业信息，并保存为JSON格式
    
    Args:
        base_path: CSV文件所在的目录路径
        columns_map: 字段映射字典，key为原始列名，value为中文列名
    """
    if not base_path:
        raise ValueError("请提供有效的路径参数")
    
    if not columns_map:
        raise ValueError("请提供有效的字段映射字典")
    
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"{base_path} 路径不存在，请检查路径是否正确")

    # 获取所有CSV文件
    csv_files = []
    base_path_obj = Path(base_path)
    
    for file_path in base_path_obj.rglob('*.csv'):
        if file_path.is_file():
            csv_files.append(file_path)
            # print(f"找到文件: {file_path.name}")
    
    if not csv_files:
        raise FileNotFoundError(f"在 {base_path} 路径下未找到任何CSV文件")

    all_data = []
    for file_path in csv_files:
        print(f"处理文件: {file_path}")
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
            # 筛选存在的列（只保留columns_map中定义的key）
            existing_cols = [col for col in columns_map.keys() if col in df.columns]
            if existing_cols:
                df_filtered = df[existing_cols]
                all_data.append(df_filtered)
                print(f"✓ 处理: {file_path.name} ({len(df_filtered)} 条记录)")
            else:
                print(f"⚠ 跳过: {file_path.name} - 未找到匹配的列")
        except Exception as e:
            print(f"✗ 跳过: {file_path.name} - {e}")

    if not all_data:
        raise ValueError("未找到有效数据")

    # 合并所有数据
    result = pd.concat(all_data, ignore_index=True)
    
    # 重命名为中文key
    result = result.rename(columns=columns_map)
    
    # 转换为JSON格式（列表形式，每个企业一个字典）
    data_list = result.to_dict(orient='records')
    
    # 保存为JSON文件
    output_file = save_path
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, ensure_ascii=False, indent=2)
    
    print(f"\n完成! 共提取 {len(data_list)} 家企业信息")
    print(f"输出文件: {output_file}")
    
    return data_list


if __name__ == '__main__':
    columns_map = {
        # 时间信息
        'Announced Date': '融资公告日期',
        # 企业基本信息
        'Organization Name': '企业名称',
        'Organization Website': '企业官网',
        'Organization Location': '企业所在地',
        # 业务概况
        'Organization Description': '企业描述',
        'Organization Industries': '所属行业',
        # 财务信息
        'Money Raised (in USD)': '本轮融资金额(美元)',
        'Total Funding Amount (in USD)': '累计融资总额(美元)',
        'Organization Revenue Range': '收入范围',
        'Funding Status': '融资状态'
    }
    extract_company_info(Config.VECTTOR_BASEDATA_PATH, columns_map)
