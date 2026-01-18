## ExectorAgent

- 长上下文
  - ExectorAgent主要承担信息源收集的任，对收集到的paper可以做一下裁剪或者摘要，方便让llm做出判断即可。但是依然要在state中保存完整的paper信息，用于clean和download。
- opnalex 和sematic_scholar 检索出来的文档无法下载
  - 工具执行的时候如果出现异常也不中断，使用asyncio.gather(*task,return_exceptions=True)的方式