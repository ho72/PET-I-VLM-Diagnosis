# handler.py 예시
from analysis import analyze_image, generate_report, create_chat_session

def handle_request(image_path_or_obj):
    # 1. 이미지 분석 (YOLO + VLM)
    # 최초 실행 시에만 모델 로딩(약간의 시간 소요), 두 번째 실행부터는 즉시 처리됨
    print("\n\n--- 1. Diagnosing ---")
    diag_result = analyze_image(image_path_or_obj)
    print("Diagnosis Result:", diag_result)

    # 2. 보고서 생성 (RAG + VLM)
    print("\n\n--- 2. Reporting ---")
    report_md = generate_report(diag_result)
    print("Report Generated (preview):", report_md, "...")

    # 3. 챗봇 세션 초기화
    print("\n\n--- 3. Init Chat ---")
    bot, case = create_chat_session(diag_result, report_md, case_id="user_123")
    
    # 4. (옵션) 바로 질문해보기
    #answer = bot.answer(case.case_id, "이 병은 어떻게 관리해?", mode="brief")
    while True:
        question = input("질문을 입력하세요(종료 - exit) : ")
        if question == "exit":
            break
        print("Chat Answer:", bot.answer(case.case_id, question, mode="brief"))
    
    # return {
    #     "diagnosis": diag_result,
    #     "report": report_md,
    #     "chat_answer": answer
    # }

# 테스트: 핸들러를 연속 두 번 호출해도 모델은 처음 한 번만 로드됩니다.
handle_request("/workspace/test_image/image/valid_blepharitis_98.jpg")
handle_request("/workspace/test_image/image/valid_Epiphora_20.jpg")