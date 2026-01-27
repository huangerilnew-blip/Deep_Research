import asyncio
import httpx

async def test_direct_download():
    """直接测试下载，看是不是 arxiv.py 代码的问题"""

    # 使用一个真实的 arXiv PDF URL
    test_pdf_url = "https://arxiv.org/pdf/2301.07041.pdf"  # 一个著名的论文

    print("=== 测试 1: 使用 httpx.get（同步）===")
    try:
        import httpx_sync
        response = httpx_sync.get(test_pdf_url, timeout=30)
        print(f"状态码: {response.status_code}")
        print(f"内容长度: {len(response.content)}")
    except Exception as e:
        print(f"错误: {e}")

    print("\n=== 测试 2: 使用 httpx.AsyncClient（和 arxiv.py 一样的方式）===")
    try:
        session = httpx.AsyncClient(timeout=30.0)
        print(f"Session 创建: {session}")

        print("开始请求...")
        response = await session.get(test_pdf_url)
        print(f"状态码: {response.status_code}")

        print("开始读取 content...")
        content = response.content
        print(f"内容长度: {len(content)}")

        await session.aclose()
        print("Session 关闭成功")
    except Exception as e:
        import traceback
        print(f"错误: {e}")
        traceback.print_exc()

    print("\n=== 测试 3: 使用 async with（推荐方式）===")
    try:
        async with httpx.AsyncClient(timeout=30.0) as session:
            print(f"Session 创建: {session}")

            print("开始请求...")
            response = await session.get(test_pdf_url)
            print(f"状态码: {response.status_code}")

            print("开始读取 content...")
            content = response.content
            print(f"内容长度: {len(content)}")

        print("Session 自动关闭成功")
    except Exception as e:
        import traceback
        print(f"错误: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_direct_download())
