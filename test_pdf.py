from langchain_community.document_loaders import PyPDFLoader

pdf_path = "/mnt/d/A_Study/Agent/Med_Agent/vector_db/清解散火颗粒中成药研发全套方案.pdf"

loader = PyPDFLoader(pdf_path)

docs = loader.load()

print("=" * 50)
print("总页数：", len(docs))
print("=" * 50)

print(docs[0].page_content[:1000])
