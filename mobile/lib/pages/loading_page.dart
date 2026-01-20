// lib/pages/loading_page.dart
import 'dart:async';
import 'dart:math';
import 'dart:ui';
import 'package:flutter/material.dart';

class LoadingPage extends StatefulWidget {
  const LoadingPage({super.key});
  @override
  State<LoadingPage> createState() => _LoadingPageState();
}

class _LoadingPageState extends State<LoadingPage>
    with SingleTickerProviderStateMixin {
  late final AnimationController _ctrl;
  late final List<_PawParticle> _particles;
  final _rng = Random(); // 매 실행마다 약간 다른 무드

  @override
  void initState() {
    super.initState();

    _ctrl = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 3600),
    )..repeat();

    // 파티클 생성
    const count = 18; // 개수 더 늘리려면 이 값만 변경
    _particles = List.generate(count, (i) {
      final p = _PawParticle();
      p.randomize(_rng, firstTime: true);
      // 서로 다른 타이밍에 리스폰되도록 속도/위상 분산
      p.speed = _lerp(0.75, 1.25, _rng.nextDouble());
      p.phase = i / count;
      return p;
    });

    // 3초 후 자동 종료 (결과 페이지로 이동할 거면 여기서 pushNamed)
  //   Timer(const Duration(seconds: 5), () {
  //     //if (mounted) Navigator.pop(context);
  //     if (mounted) Navigator.pushNamed(context, '/result');
  //   });
  }

  @override
  void dispose() {
    _ctrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Stack(
      children: [
        // 은은한 비네트
        IgnorePointer(
          child: Container(
            decoration: const BoxDecoration(
              gradient: RadialGradient(
                center: Alignment(0, -0.1),
                radius: 1.2,
                colors: [Colors.transparent, Colors.transparent, Color(0x10A07E47)],
                stops: [0.0, 0.7, 1.0],
              ),
            ),
          ),
        ),

        // 파티클 렌더
        AnimatedBuilder(
          animation: _ctrl,
          builder: (_, __) {
            final base = _ctrl.value; // 0..1

            return Stack(
              children: _particles.map((p) {
                // 각 파티클의 개별 진행도
                final prog = (base * p.speed + p.phase) % 1.0;

                // 래핑 감지: 새 사이클 진입 시 즉시 재랜덤(= 아래에서 새로 스폰)
                if (prog < p.prevProg) {
                  p.randomize(_rng);
                }
                p.prevProg = prog;

                return _PawRise(progress: prog, conf: p);
              }).toList(),
            );
          },
        ),

        // 중앙 텍스트 (펄스 + 점 순환)
        _LoadingCenterText(controller: _ctrl),
      ],
    );
  }

  static double _lerp(double a, double b, double t) => a + (b - a) * t;
}

// 중앙 텍스트
class _LoadingCenterText extends StatelessWidget {
  const _LoadingCenterText({required this.controller});
  final AnimationController controller;

  @override
  Widget build(BuildContext context) {
    final t = CurvedAnimation(parent: controller, curve: Curves.easeInOut).value;
    final scale = 1.0 + (sin(t * pi * 2) * 0.02); // ±2%
    final dots = 1 + (controller.value * 3).floor() % 3;

    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          SizedBox(
            child: Image.asset('assets/logo_img.png', width: 300),
          ),
          SizedBox(height: 30),
          Container(
            child: Text(
              '진단 중' + '...',
              style: const TextStyle(
                fontSize: 28,
                fontWeight: FontWeight.w700,
                color: Colors.black87,
              ),
            ),
          ),
          SizedBox(height: 30),
          Container(
            child: Text(
              'Pet-I가 열심히 진단 중이에요!'
            ),
          ),
        ],
      ),
    );
  }
}

/// 파티클 상태 (한 사이클 끝날 때마다 재랜덤)
class _PawParticle {
  // 렌더링 속성
  double leftFactor = 0.5;   // 0..1
  double size = 36;          // px
  double rotationDeg = 0;    // 회전
  double blurSigma = 0;      // 블러
  double opacityBase = 0.35; // 기본 불투명

  // 애니메이션 제어
  double speed = 1.0;        // 개별 속도 배수(0.75~1.25 정도)
  double phase = 0.0;        // 시작 오프셋
  double prevProg = 0.0;     // 래핑 감지용

  void randomize(Random rng, {bool firstTime = false}) {
    // 새로 스폰될 때마다 속성 변경(자연스러운 다양성)
    leftFactor = rng.nextDouble().clamp(0.08, 0.92);
    size = 30 + rng.nextInt(14).toDouble();           // 30~43
    rotationDeg = rng.nextDouble() * 24 - 12;         // -12~12도
    blurSigma = rng.nextDouble() * 1.6;               // 0.0~1.6
    opacityBase = 0.28 + rng.nextDouble() * 0.17;     // 0.28~0.45

    if (firstTime) {
      prevProg = 0.0;
    }
  }
}

/// 발바닥 하나: 아래(화면 밖) → 위(화면 밖) 곡선 이동
class _PawRise extends StatelessWidget {
  const _PawRise({required this.progress, required this.conf});
  final double progress; // 0..1
  final _PawParticle conf;

  static const Color pawColor = Color(0xFFDCC7A6); // 배경 FF FB F4와 어울림

  @override
  Widget build(BuildContext context) {
    final t = Curves.easeInOut.transform(progress);

    // 세로: 아래(1.18) → 위(-1.08)
    final yAlign = _lerp(1.18, -1.08, t);

    // 좌우 곡선
    final sway = sin(t * pi) * 16.0;

    // 시작/끝에서 0, 중간에서 가장 진함 (랩 순간 안 보이게)
    final opacity = (1.0 - (t - 0.5).abs() * 2).clamp(0.0, 1.0) * conf.opacityBase;

    // 회전 라디안
    final rot = conf.rotationDeg * pi / 180.0;

    return Positioned.fill(
      child: IgnorePointer(
        child: Opacity(
          opacity: opacity,
          child: Align(
            alignment: Alignment(_toAlign(conf.leftFactor), yAlign),
            child: Transform.translate(
              offset: Offset(sway, 0),
              child: Transform.rotate(
                angle: rot,
                child: ImageFiltered(
                  imageFilter: conf.blurSigma > 0
                      ? ImageFilter.blur(sigmaX: conf.blurSigma, sigmaY: conf.blurSigma)
                      : ImageFilter.blur(sigmaX: 0, sigmaY: 0),
                  child: Icon(Icons.pets, size: conf.size, color: pawColor),
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }

  static double _toAlign(double t) => t * 2 - 1;
  static double _lerp(double a, double b, double t) => a + (b - a) * t;
}
