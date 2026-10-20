from schemas import retrieval, isUse

query = retrieval(
    retrieval_query="Navnit professional experience employers"
)

grade = isUse(
    isuse="useful",
    reason="Correctly states that employment history is undocumented."
)

print(query.model_dump())
print(grade.model_dump())
print("Schema validation passed.")