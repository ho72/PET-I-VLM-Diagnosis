import 'package:flutter/material.dart';
import 'package:flutter_markdown/flutter_markdown.dart';
import 'dart:typed_data';

const String kDummyMarkdown = '''
# 분석 대기 중
데이터를 불러오는 중이거나 분석 결과가 없습니다.
''';

class DiagResult extends StatelessWidget {
  const DiagResult({super.key});

  @override
  Widget build(BuildContext context) {
    final Color primaryColor = const Color(0xFF4A90E2);
    final Color cardColor = Colors.white;
    // final Color warningColor = const Color(0xFFFFF3E0); // 경고문구 색상
    final TextTheme textTheme = Theme.of(context).textTheme;

    final args = ModalRoute.of(context)?.settings.arguments;
    
    // 데이터 변수 초기화
    String markdown = kDummyMarkdown;
    String petName = "이름 미입력";
    Uint8List? imageBytes;
    String? caseId;
    
    // 🔹 진단명과 증상 파싱을 위한 변수
    String? diagnosisTitle;
    List<String> symptomsList = [];

    if (args is Map) {
      final argMd = args['markdown'] as String?;
      if (argMd != null && argMd.isNotEmpty) {
        markdown = argMd;
      }
      final argName = args['name'] as String?;
      if (argName != null) {
        petName = argName;
      }
      final argImageBytes = args['imageBytes'] as Uint8List?;
      if (argImageBytes != null) {
        imageBytes = argImageBytes;
      }
      final argCaseId = args['caseId'] as String?;
      if (argCaseId != null) {
        caseId = argCaseId;
      }
      
      // 🔹 진단 결과 객체(JSON) 파싱
      final diagData = args['diagnosis'];
      if (diagData is Map) {
        diagnosisTitle = diagData['diagnosis']?.toString();
        if (diagData['symptoms'] is List) {
          symptomsList = (diagData['symptoms'] as List)
              .map((e) => e.toString())
              .toList();
        }
      }
    } else if (args is String) {
      markdown = args;
    }

    return SafeArea(
      child: Center(
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: 500),
          child: SingleChildScrollView(
            padding: const EdgeInsets.symmetric(horizontal: 16.0, vertical: 24.0),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                // 1. 상단 네비게이션 (홈 아이콘)
                _buildTopBar(context, caseId, diagData: args is Map ? args['diagnosis'] : null),

                const SizedBox(height: 24),

                // 2. 이미지 & 이름 카드
                _buildProfileCard(
                  cardColor,
                  textTheme,
                  petName,
                  caseId,
                  imageBytes,
                ),
                
                const SizedBox(height: 16),

                // 3. [핵심] 진단명 & 증상 요약 카드 (새로 추가된 부분)
                if (diagnosisTitle != null) 
                  _buildDiagnosisSummaryCard(cardColor, textTheme, diagnosisTitle, symptomsList),

                // 진단명이 없는 경우(에러 등)에는 표시 안 함
                if (diagnosisTitle != null)
                  const SizedBox(height: 16),

                // 4. 진단 보고서 (마크다운)
                const Padding(
                  padding: EdgeInsets.only(left: 4, bottom: 8),
                  child: Text(
                    '상세 리포트',
                    style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold, color: Colors.black54),
                  ),
                ),
                _buildDetailsCard(cardColor, textTheme, markdown),
                
                const SizedBox(height: 32),

                // 5. 하단 액션 버튼들
                _buildActionButtons(context, caseId),
                const SizedBox(height: 20),
              ],
            ),
          ),
        ),
      ),
    );
  }

  // 상단 네비게이션 바
  Widget _buildTopBar(BuildContext context, String? caseId, {dynamic diagData}) {
    return Row(
      children: [
        IconButton(
          icon: const Icon(Icons.home_outlined, color: Colors.black87, size: 28),
          splashRadius: 22,
          onPressed: () {
            // 홈으로 갈 때 현재 진단 데이터를 넘겨줄지 여부 결정
            if (caseId == null) {
              Navigator.popUntil(context, ModalRoute.withName('/start'));
            } else {
              Navigator.pushNamedAndRemoveUntil(
                context,
                '/start',
                (route) => false,
                arguments: {
                  'caseId': caseId,
                  'diagnosis': diagData,
                },
              );
            }
          },
        ),
        Expanded(
          child: Center(
            child: Image.asset(
              'assets/logo_img.png',
              height: 20,
            ),
          ),
        ),
        const SizedBox(width: 48), // 아이콘 균형 맞추기
      ],
    );
  }

  // 1번 영역: 사진 + 이름 + CaseID
  Widget _buildProfileCard(Color cardColor, TextTheme textTheme, String petName, String? caseId, Uint8List? imageBytes) {
    return Card(
      elevation: 0, // 깔끔하게 그림자 제거 (원하시면 숫자를 높이세요)
      color: cardColor,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(16),
        side: BorderSide(color: Colors.grey.shade200, width: 1), // 얇은 테두리
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // 이미지 영역
          ClipRRect(
            borderRadius: const BorderRadius.vertical(top: Radius.circular(16)),
            child: imageBytes != null
                ? Image.memory(
                    imageBytes,
                    width: double.infinity,
                    height: 220, // 사진 높이 조금 키움
                    fit: BoxFit.cover,
                  )
                : Container(
                    width: double.infinity,
                    height: 220,
                    color: Colors.grey[200],
                    child: const Icon(Icons.pets, size: 60, color: Colors.grey),
                  ),
          ),
          
          // 텍스트 영역
          Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start, // 좌측 정렬
              children: [
                Text(
                  '반려동물 이름',
                  style: TextStyle(fontSize: 12, color: Colors.grey[600]),
                ),
                const SizedBox(height: 4),
                Text(
                  petName,
                  style: textTheme.headlineSmall?.copyWith(
                    fontWeight: FontWeight.bold,
                    color: Colors.black87,
                  ),
                ),
                if (caseId != null) ...[
                  const SizedBox(height: 8),
                  Text(
                    'Case ID: $caseId',
                    style: TextStyle(fontSize: 11, color: Colors.grey[400]),
                  ),
                ]
              ],
            ),
          ),
        ],
      ),
    );
  }

  // 2번 영역: 진단명 + 증상 리스트 (핵심)
  Widget _buildDiagnosisSummaryCard(Color cardColor, TextTheme textTheme, String? title, List<String> symptoms) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: cardColor,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: const Color(0xFF4A90E2).withOpacity(0.3), width: 1.5), // 강조 테두리
        boxShadow: [
          BoxShadow(
            color: const Color(0xFF4A90E2).withOpacity(0.08),
            blurRadius: 12,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start, // 좌측 정렬
        children: [
          const Text(
            '진단 결과',
            style: TextStyle(fontSize: 13, fontWeight: FontWeight.w600, color: Color(0xFF4A90E2)),
          ),
          const SizedBox(height: 6),
          // 진단명 (크게)
          Text(
            title ?? '분석 중...',
            style: textTheme.headlineMedium?.copyWith(
              fontWeight: FontWeight.w800,
              color: Colors.black87,
              height: 1.2,
            ),
          ),
          const SizedBox(height: 16),
          
          // 증상 리스트 (Divider로 구분)
          if (symptoms.isNotEmpty) ...[
            const Divider(height: 1, thickness: 1, color: Color(0xFFEEEEEE)),
            const SizedBox(height: 16),
            const Text(
              '발견된 주요 증상',
              style: TextStyle(fontSize: 13, fontWeight: FontWeight.w600, color: Colors.black54),
            ),
            const SizedBox(height: 10),
            Wrap(
              spacing: 8.0, // 가로 간격
              runSpacing: 8.0, // 세로 간격
              children: symptoms.map((symptom) {
                return Container(
                  padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                  decoration: BoxDecoration(
                    color: const Color(0xFFF5F7FA), // 연한 회색 배경
                    borderRadius: BorderRadius.circular(8),
                    border: Border.all(color: Colors.grey.shade300),
                  ),
                  child: Text(
                    symptom,
                    style: const TextStyle(
                      fontSize: 14,
                      color: Colors.black87,
                      fontWeight: FontWeight.w500,
                    ),
                  ),
                );
              }).toList(),
            ),
          ]
        ],
      ),
    );
  }

  // 3번 영역: 마크다운 상세 보고서
  Widget _buildDetailsCard(Color cardColor, TextTheme textTheme, String markdown) {
    return Card(
      elevation: 0,
      color: cardColor,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(16),
        side: BorderSide(color: Colors.grey.shade200),
      ),
      child: Padding(
        padding: const EdgeInsets.all(20.0),
        child: MarkdownBody(
          data: markdown,
          styleSheet: MarkdownStyleSheet(
            // 텍스트 스타일 정의 (marginTop 삭제함)
            h1: textTheme.titleLarge?.copyWith(fontWeight: FontWeight.bold, height: 1.5, color: Colors.black87),
            h2: textTheme.titleMedium?.copyWith(fontWeight: FontWeight.bold, height: 1.5, color: Colors.black87),
            p: textTheme.bodyMedium?.copyWith(height: 1.6, color: Colors.black87),
            listBullet: textTheme.bodyMedium?.copyWith(color: Colors.black54),
            strong: const TextStyle(fontWeight: FontWeight.bold, color: Color(0xFF4A90E2)),
            
            // ✅ 여백은 여기서 설정해야 합니다
            h2Padding: const EdgeInsets.only(top: 20, bottom: 8), 
            pPadding: const EdgeInsets.only(bottom: 8),
          ),
        ),
      ),
    );
  }
  
  // 하단 버튼들
  Widget _buildActionButtons(BuildContext context, String? caseId) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        Expanded(
          child: ElevatedButton(
            style: ElevatedButton.styleFrom(
              backgroundColor: const Color(0xFF4A90E2),
              foregroundColor: Colors.white,
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
              padding: const EdgeInsets.symmetric(vertical: 14),
              elevation: 2,
            ),
            onPressed: () {
              if (caseId == null){
                ScaffoldMessenger.of(context).showSnackBar(
                  const SnackBar(content: Text('진단 기록이 저장되지 않았습니다.')),
                );                         
                return; 
              }
              Navigator.pushNamed(context, '/chatbot', arguments: {
                'caseId': caseId,
              });
            },
            child: Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: const [
                 Icon(Icons.chat_bubble_outline, size: 20),
                 SizedBox(width: 8),
                 Text('AI 상담하기', style: TextStyle(fontSize: 15, fontWeight: FontWeight.bold)),
              ],
            ),
          ),
        ),
        // 필요 시 다른 버튼 추가 (수의사 상담 등)
      ],
    );
  }
}