import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:learn_flutter/pages/diag_result_page.dart';
import 'pages/diag_start_page.dart';
import 'pages/diag_page.dart';
import 'pages/loading_page.dart';
import 'pages/chat_page.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  // 여기서 전역으로 프레임(폭 제한 + 축소/확대)을 적용
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: true,
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF4A90E2),
          background: const Color(0xFFFFFBF4),
        ),
        textTheme: GoogleFonts.notoSansKrTextTheme(),
        elevatedButtonTheme: ElevatedButtonThemeData(
          style: ElevatedButton.styleFrom(
            elevation: 2,
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(12),
            ),
            padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
          ),
        ),
        cardTheme: CardThemeData(
          elevation: 2,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(16),
          ),
          color: Colors.white,
        ),
      ),

      builder: (context, child) => Scaffold(
        backgroundColor: _ResponsiveFrame.kBackgroundColor,
        body: _ResponsiveFrame(child: child ?? const SizedBox()),
      ),
      initialRoute: '/start',
      routes: {
        '/start': (_) => const DiagStart(), // <- 페이지들은 Scaffold 없이 '바디 위젯'만
        '/diag':  (_) => const DiagPage(),  // <- 바디 위젯만
        '/loading': (_) => const LoadingPage(),
        '/result' : (_) => const DiagResult(),
        '/chatbot' : (_) => const ChatPage(),
      },

      // 모든 라우트(페이지)를 _ResponsiveFrame으로 감싸기
      // builder: (context, child) => _ResponsiveFrame(child: child ?? const SizedBox()),
      //
      // initialRoute: '/start',
      // routes: {
      //   '/start': (_) => const DiagStart(),
      //   '/diag':  (_) => const DiagPage(),
      // },
    );
  }
}


// 화면 크기를 관리하는 공용 프레임:
// - 너무 클 때는 maxWidth로 고정하여 그대로 사용
// - 보통 크기에서는 가용 너비를 그대로 사용
// - 너무 작아지면 (FHD 이하 모바일 등) 자연스럽게 '축소 렌더링'하여 레이아웃 붕괴 방지
class _ResponsiveFrame extends StatelessWidget {
  const _ResponsiveFrame({super.key, required this.child});

  final Widget child;

  // 필요시 숫자 변경
  static const double kMinDesignWidth = 420;  // 이 폭을 기준으로 '축소' 시작
  static const double kMaxContentWidth = 1100;  // 이 이상은 더 안키움 (가운데 정렬)

  static const Color kBackgroundColor = Color(0xFFFFFBF4);  // 또는 FAF0E6

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, constraints) {
        final double vw = constraints.maxWidth;
        final double vh = constraints.maxHeight;

        Widget framed = child;

        // 1) 너무 큰 화면 : 가운데 정렬 + 최대 폭 고정
        if (vw > kMaxContentWidth) {
          framed = Center(
            child: SizedBox(
              width: kMaxContentWidth,
              height: vh,
              child: child,
            ),
          );
        }

        // 2) 보통 화면 : 가용 폭 사용 (여백은 페이지 내부에서 결정)
        else if (vw >= kMinDesignWidth) {
          framed = SizedBox(
            width: vw,
            height: vh,
            child: child,
          );
        }

        // 3) 너무 작은 화면 : '축소 렌더링'
        // - FittedBox로 kMinDesignWidth 기준 레이아웃을 비율 축소
        // - 글자/아이콘이 너무 작아질 수 있으니 페이지 내 요소도 적절히 대응하도록 설계 권장
        else {
          framed = Center(
            child: FittedBox(
              fit: BoxFit.contain,
              alignment: Alignment.topCenter,
              child: SizedBox(
                width: kMinDesignWidth,
                height: vh,
                child: child,
              ),
            ),
          );
        }

        // 프레임은 레이아웃/배경만 담당
        return ColoredBox(
          color: kBackgroundColor,
          child: SafeArea(child: framed),
        );
      },
    );
  }
}



