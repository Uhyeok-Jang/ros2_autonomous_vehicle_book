// 최대 입력 문자 수
#include <stdlib.h>

const unsigned int MAX_INPUT = 15;

// 핀 번호 변수
const int STEERING_1 = 2;
const int STEERING_2 = 3;
const int FORWARD_RIGHT_1 = 5;
const int FORWARD_RIGHT_2 = 4;
const int FORWARD_LEFT_1 = 6;
const int FORWARD_LEFT_2 =7;
const int POT = A2;

// 조향 속도 상수
const int STEERING_SPEED = 128;

// 가변저항 값 범위
const int resistance_most_left = 152;
const int resistance_most_right = 64;

// 조향 최대 단계 수 (한 쪽 기준)
const int MAX_STEERING_STEP = 7;

// 제어 상태 변수
int angle = 0, resistance = 0, mapped_resistance = 0;
int left_speed = 0, right_speed = 0;

// 명령 주기 제한 변수
unsigned long lastCommandTime = 0; // 마지막 명령 처리 시간
const unsigned int COMMAND_INTERVAL = 50; // 명령 처리 간 최소 대기 시간(ms)
// Only discard interrupted input fragments; this does NOT stop motor outputs.
const unsigned long INPUT_FRAGMENT_TIMEOUT_MS = 500;
bool commandActive = false;

// 함수 선언
void steerRight();
void steerLeft();
void maintainSteering();
void setLeftMotorSpeed(int speed);
void setRightMotorSpeed(int speed);
void processIncomingByte(const byte inByte);
void processData(const char *data);
void stopVehicle();
bool parseField(const char *&cursor, char marker, long &value);

void setup() {
    Serial.begin(9600);

    // 핀 모드 설정
    pinMode(POT, INPUT);
    pinMode(STEERING_1, OUTPUT);
    pinMode(STEERING_2, OUTPUT);
    pinMode(FORWARD_RIGHT_1, OUTPUT);
    pinMode(FORWARD_RIGHT_2, OUTPUT);
    pinMode(FORWARD_LEFT_1, OUTPUT);
    pinMode(FORWARD_LEFT_2, OUTPUT);
    stopVehicle(); // Remain disarmed until a complete valid command arrives.
}

void loop() {
    // 현재 시간 가져오기
    unsigned long currentTime = millis();

    // 직렬 데이터 처리
    while (Serial.available() > 0) {
        processIncomingByte(Serial.read());
    }

    currentTime = millis();
    // Remain stopped at startup or after an explicit zero command.
    // No automatic motor cutoff when command reception pauses.
    if (!commandActive) {
        stopVehicle();
        return;
    }

    // 일정 시간 간격으로만 제어 명령 실행
    if (currentTime - lastCommandTime >= COMMAND_INTERVAL) {
        // 포텐셔미터 값을 읽어 조향 계산
        resistance = analogRead(POT);
        mapped_resistance = map(resistance, resistance_most_left, resistance_most_right, -MAX_STEERING_STEP, MAX_STEERING_STEP + 1);

        // 조향 상태에 따라 동작 제어
        if (mapped_resistance == angle) {
            maintainSteering();
        } else if (mapped_resistance > angle) {
            steerLeft();
        } else {
            steerRight();
        }

        // 모터 속도 설정
        setLeftMotorSpeed(left_speed);
        setRightMotorSpeed(right_speed);

        // 마지막 명령 시간 갱신
        lastCommandTime = currentTime;
    }
}

void stopVehicle() {
    commandActive = false;
    angle = 0;
    left_speed = 0;
    right_speed = 0;
    setLeftMotorSpeed(0);
    setRightMotorSpeed(0);
    maintainSteering(); // Disable steering output too; do not recenter on a stop.
}

// 조향 제어 함수
void steerRight() {
    analogWrite(STEERING_1, STEERING_SPEED);
    analogWrite(STEERING_2, LOW);
}

void steerLeft() {
    analogWrite(STEERING_1, LOW);
    analogWrite(STEERING_2, STEERING_SPEED);
}

void maintainSteering() {
    analogWrite(STEERING_1, LOW);
    analogWrite(STEERING_2, LOW);
}

// 모터 속도 설정 함수
void setLeftMotorSpeed(int speed) {
    if (speed > 0) {
        analogWrite(FORWARD_LEFT_1, speed);
        analogWrite(FORWARD_LEFT_2, LOW);
    } else {
        analogWrite(FORWARD_LEFT_1, LOW);
        analogWrite(FORWARD_LEFT_2, (-1) * speed);
    }
}

void setRightMotorSpeed(int speed) {
    if (speed > 0) {
        analogWrite(FORWARD_RIGHT_1, speed);
        analogWrite(FORWARD_RIGHT_2, LOW);
    } else {
        analogWrite(FORWARD_RIGHT_1, LOW);
        analogWrite(FORWARD_RIGHT_2, (-1) * speed);
    }
}

// 직렬 데이터 처리
void processIncomingByte(const byte inByte) {
    static char input_line[MAX_INPUT];
    static unsigned int input_pos = 0;
    static bool input_overflow = false;
    static unsigned long lastByteTime = 0;
    const unsigned long now = millis();
    if (now - lastByteTime >= INPUT_FRAGMENT_TIMEOUT_MS) {
        input_pos = 0; // Do not later accept an old, interrupted partial packet.
        input_overflow = false;
    }
    lastByteTime = now;

    switch (inByte) {
        case '\n':
            input_line[input_pos] = 0; // 종료 문자 추가
            if (!input_overflow) processData(input_line);
            input_pos = 0; // 버퍼 초기화
            input_overflow = false;
            break;

        case '\r':
            break; // 캐리지 리턴 무시

        default:
            if (input_pos < (MAX_INPUT - 1)) {
                input_line[input_pos++] = inByte;
            } else {
                input_overflow = true; // Discard the WHOLE oversized frame.
            }
            break;
    }
}

bool parseField(const char *&cursor, char marker, long &value) {
    if (*cursor != marker) return false;
    ++cursor;
    const char *digits = cursor;
    if (*digits == '-' || *digits == '+') ++digits;
    if (*digits < '0' || *digits > '9') return false;
    char *end = NULL;
    value = strtol(cursor, &end, 10);
    cursor = end;
    return true;
}

// Only complete, in-range s...l...r... packets update the command.
void processData(const char *data) {
    const char *cursor = data;
    long newAngle, newLeftSpeed, newRightSpeed;
    if (!parseField(cursor, 's', newAngle)
        || !parseField(cursor, 'l', newLeftSpeed)
        || !parseField(cursor, 'r', newRightSpeed) || *cursor != '\0') return;
    if (newAngle < -MAX_STEERING_STEP || newAngle > MAX_STEERING_STEP
        || newLeftSpeed < -255 || newLeftSpeed > 255
        || newRightSpeed < -255 || newRightSpeed > 255) return;

    angle = (int)newAngle;
    left_speed = (int)newLeftSpeed;
    right_speed = (int)newRightSpeed;
    commandActive = angle != 0 || left_speed != 0 || right_speed != 0;
    if (!commandActive) stopVehicle(); // s0l0r0 stops all outputs immediately.
}
