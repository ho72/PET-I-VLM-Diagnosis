// lib/pages/diag_page.dart
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart'; // ✅ 교체
import 'dart:typed_data';
import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:http_parser/http_parser.dart';


class DiagPage extends StatefulWidget {
  const DiagPage({super.key});
  @override
  State<DiagPage> createState() => _DiagPageState();
}

class _DiagPageState extends State<DiagPage> {
  final _nameCtrl = TextEditingController();
  final _ageCtrl  = TextEditingController();
  Uint8List? _pickedBytes;

  final _picker = ImagePicker(); // ✅ image_picker 인스턴스

  Future<void> _pickImage() async {
    // 갤러리에서 이미지 선택 (웹에서도 동작)
    final XFile? xfile = await _picker.pickImage(
      source: ImageSource.gallery,
      maxWidth: 2048, // 선택: 리사이즈 (웹/모바일 공통)
      maxHeight: 2048,
      imageQuality: 90, // 선택: JPEG 압축 (0~100)
    );
    if (xfile != null) {
      final bytes = await xfile.readAsBytes(); // ✅ 웹/모바일 공통
      setState(() => _pickedBytes = bytes);
    }
  }

  @override
  void dispose() {
    _nameCtrl.dispose();
    _ageCtrl.dispose();
    super.dispose();
  }

  InputDecoration _inputDecoration() => InputDecoration(
    isDense: true,
    filled: true,
    fillColor: Colors.white,
    hintText: '',
    contentPadding: const EdgeInsets.symmetric(horizontal: 14, vertical: 14),
    border: OutlineInputBorder(
      borderRadius: BorderRadius.circular(10),
      borderSide: BorderSide(color: Colors.black.withOpacity(0.15), width: 1),
    ),
    enabledBorder: OutlineInputBorder(
      borderRadius: BorderRadius.circular(10),
      borderSide: BorderSide(color: Colors.black.withOpacity(0.15), width: 1),
    ),
    focusedBorder: OutlineInputBorder(
      borderRadius: BorderRadius.circular(10),
      borderSide: const BorderSide(color: Color(0xFF4A90E2), width: 1.2),
    ),
  );

  @override
  Widget build(BuildContext context) {
    final vh = MediaQuery.of(context).size.height;
    const maxInnerWidth = 420.0;

    return Center(
      child: SingleChildScrollView(
        child: ConstrainedBox(
          constraints: BoxConstraints(minHeight: vh, maxWidth: maxInnerWidth),
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                // 상단 뒤로가기
                Row(
                  children: [
                    IconButton(
                      icon: const Icon(Icons.arrow_back_ios_new),
                      splashRadius: 22,
                      onPressed: () => Navigator.pop(context),
                    ),

                    Expanded(
                      child: Center(
                        child: Image.asset(
                          'assets/logo_img.png',
                          height: 20,
                        ),
                      ),
                    ),
                    const SizedBox(width: 48),
                  ],
                ),
                const SizedBox(height: 50),

                const Text('이름', style: TextStyle(fontSize: 18, fontWeight: FontWeight.w600)),
                const SizedBox(height: 8),
                TextFormField(
                  controller: _nameCtrl,
                  decoration: _inputDecoration().copyWith(hintText: '이름을 입력하세요'),
                ),
                const SizedBox(height: 18),

                // const Text('나이', style: TextStyle(fontSize: 18, fontWeight: FontWeight.w600)),
                // const SizedBox(height: 8),
                // TextFormField(
                //   controller: _ageCtrl,
                //   keyboardType: TextInputType.number,
                //   decoration: _inputDecoration().copyWith(hintText: '나이를 입력하세요'),
                // ),
                // const SizedBox(height: 22),

                // 업로드 박스
                GestureDetector(
                  onTap: _pickImage, // ✅ image_picker 사용
                  child: Container(
                    width: double.infinity,
                    constraints: const BoxConstraints(minHeight: 220),
                    padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 16),
                    decoration: BoxDecoration(
                      color: Colors.white,
                      borderRadius: BorderRadius.circular(12),
                      border: Border.all(
                        color: Colors.black.withOpacity(0.18),
                        width: 1,
                      ),
                    ),
                    child: _pickedBytes == null
                        ? Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        const Icon(Icons.photo_camera_outlined, size: 50, color: Colors.black54),
                        const SizedBox(height: 10),
                        Text('안구 사진 업로드',
                            style: TextStyle(fontSize: 14, color: Colors.black.withOpacity(0.65))),
                        const SizedBox(height: 4),
                        Text('클릭하여 선택 또는 촬영',
                            style: TextStyle(fontSize: 12, color: Colors.black.withOpacity(0.45))),
                      ],
                    )
                        : ClipRRect(
                      borderRadius: BorderRadius.circular(8),
                      child: Image.memory(
                        _pickedBytes!,
                        fit: BoxFit.cover,
                        height: 260,
                        width: double.infinity,
                      ),
                    ),
                  ),
                ),

                const SizedBox(height: 24),
                const SizedBox(height: 30),

                // 진단하기 버튼
                SizedBox(
                  width: double.infinity,
                  height: 52,
                  child: ElevatedButton(
                    style: ElevatedButton.styleFrom(
                      backgroundColor: const Color(0xFF4A90E2),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(8),
                      ),
                      elevation: 0,
                    ),
                    onPressed: () async {
                      if (_pickedBytes == null) {
                        ScaffoldMessenger.of(context).showSnackBar(
                          const SnackBar(content: Text('안구 사진을 먼저 업로드해주세요.')),
                        );
                        return;
                      }

                      // 화면에 보여줄 값들 미리 저장
                      final Uint8List imageBytes = _pickedBytes!;
                      final String petName = _nameCtrl.text;

                      // 1) 먼저 로딩 페이지로 이동
                      Navigator.pushNamed(context, '/loading');

                      try {
                        // 2) 로딩 페이지가 떠 있는 동안 백엔드에 진단 요청
                        final result = await _requestDiagnosis(imageBytes);
                        final markdown  = result['markdown'] as String? ?? '';
                        final caseId    = result['caseId'] as String?;
                        final diagnosis = result['diagnosis'];

                        if (!context.mounted) return;

                        // 3) 로딩 페이지 닫기
                        Navigator.pop(context);

                        // 4) 결과 페이지로 이동 (마크다운 + 이름 + 이미지 전달)
                        Navigator.pushNamed(
                          context,
                          '/result',
                          arguments: {
                            'markdown': markdown,
                            'name': petName,
                            'imageBytes': imageBytes,
                            'caseId': caseId,
                            'diagnosis': diagnosis,
                          },
                        );
                      } catch (e) {
                        if (!context.mounted) return;

                        // 에러가 나도 로딩 페이지는 닫아주기
                        Navigator.pop(context);

                        ScaffoldMessenger.of(context).showSnackBar(
                          SnackBar(content: Text('진단 요청 실패: $e')),
                        );
                      }
                    },
                    child: const Text(
                      '진단하기',
                      style: TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.w600,
                        color: Colors.white,
                      ),
                    ),
                  ),
                ),
                const SizedBox(height: 8),
              ],
            ),
          ),
        ),
      ),
    );

  }
  // ✅ 백엔드 호출 함수
  Future<Map<String, dynamic>> _requestDiagnosis(Uint8List imageBytes) async {
    // - 웹: http://localhost:8000/predict
    // - 안드로이드 에뮬: http://10.0.2.2:8000/predict
    final uri = Uri.parse('http://localhost:8000/predict');

    final request = http.MultipartRequest('POST', uri)
      ..files.add(
        http.MultipartFile.fromBytes(
          'image',
          imageBytes,
          filename: 'eye.jpg',
          contentType: MediaType('image', 'jpeg'),
        ),
      )
      ..fields['case_id'] = _nameCtrl.text; // 원하면 이름/유저ID로 case_id 지정도 가능

    final streamed = await request.send();
    final response = await http.Response.fromStream(streamed);

    if (response.statusCode != 200) {
      throw Exception('Backend error: ${response.statusCode}');
    }

    print('RAW BODY = ${response.body}');
    final body = jsonDecode(response.body);
    final data = body['data'] ?? {};

    final markdown   = (data['report_markdown'] ?? '') as String;
    final caseId     = data['case_id'];
    final diagnosis  = data['diagnosis'];

    return {
      'markdown': markdown.trim(),
      'caseId': caseId,
      'diagnosis': diagnosis,
    };
  }
}
