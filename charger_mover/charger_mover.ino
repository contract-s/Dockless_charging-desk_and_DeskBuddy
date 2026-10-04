/*
  charger_mover.ino  —  ESP32 firmware for the crossed-rod MagSafe mover.

  Mechanism:
    X motor = TOP track   -> moves the VERTICAL rod left/right   -> charger X
    Y motor = LEFT track  -> moves the HORIZONTAL rod up/down    -> charger Y
    The MagSafe charger is fixed where the rods cross, so it sits at (X, Y).

  The laptop (motion.py) does all the maths (phone cm -> motor steps). This board just
  moves each motor to the absolute step count it is told, with smooth acceleration.

  Serial protocol, 115200 baud, one command per line, every command answers "ok" or "error: ..."
    G <x_steps> <y_steps>   move both motors to these absolute step positions
    Z                       call the current position 0,0 (put the charger at home by hand first)
    H                       home with limit switches (only if USE_LIMIT_SWITCHES is 1)
    S                       stop (decelerate)
    ?                       status:  "pos <x> <y> moving <0|1>"
    E <0|1>                 motors off / on (off lets you push the rods by hand)

  Library: AccelStepper (Arduino IDE: Tools > Manage Libraries > "AccelStepper" by Mike McCauley)
  Board:   ESP32 Dev Module  (ELEGOO ESP-WROOM-32)
*/

#include <AccelStepper.h>

// ============================== SETTINGS ==============================
// Which motors/drivers you have:
//   1 = NEMA17 (or any bipolar stepper) with a STEP/DIR driver: A4988, DRV8825, TMC2209
//   2 = 28BYJ-48 with a ULN2003 driver board (the small 5 V geared motor with a white plug)
#define MOTOR_TYPE 1

// Max speed and acceleration in steps/s and steps/s^2.
#if MOTOR_TYPE == 1
const float MAX_SPEED = 4000;    // 1/16 microstepping, GT2 20T: 4000 steps/s = 50 mm/s
const float ACCEL     = 3000;
#else
const float MAX_SPEED = 600;     // 28BYJ-48 cannot go much faster than this (~15 RPM)
const float ACCEL     = 400;
#endif

// Pins. Change to match your wiring.
#if MOTOR_TYPE == 1
const int X_STEP = 26, X_DIR = 27;   // top track motor (charger left/right)
const int Y_STEP = 25, Y_DIR = 33;   // left track motor (charger up/down)
const int EN_PIN = 13;               // driver ENABLE (active LOW), shared; -1 if not wired
#else
// ULN2003 IN1..IN4.  AccelStepper wants them in the order IN1, IN3, IN2, IN4 for the 28BYJ-48.
const int X_IN1 = 26, X_IN2 = 25, X_IN3 = 33, X_IN4 = 32;
const int Y_IN1 = 23, Y_IN2 = 22, Y_IN3 = 21, Y_IN4 = 19;
#endif

// Optional limit switches at the HOME end of each track (switch to GND, closed = pressed).
#define USE_LIMIT_SWITCHES 0
const int X_LIMIT = 4, Y_LIMIT = 16;
const float HOME_SPEED = 400;        // steps/s while searching for the switch
const long  HOME_MAX_STEPS = 60000;  // give up after this many steps
const long  HOME_BACKOFF = 200;      // steps to move off the switch after touching it
// ======================================================================

#if MOTOR_TYPE == 1
AccelStepper motorX(AccelStepper::DRIVER, X_STEP, X_DIR);
AccelStepper motorY(AccelStepper::DRIVER, Y_STEP, Y_DIR);
#else
AccelStepper motorX(AccelStepper::FULL4WIRE, X_IN1, X_IN3, X_IN2, X_IN4);
AccelStepper motorY(AccelStepper::FULL4WIRE, Y_IN1, Y_IN3, Y_IN2, Y_IN4);
#endif

String line;
bool motorsOn = true;

void setMotors(bool on) {
  motorsOn = on;
#if MOTOR_TYPE == 1
  if (EN_PIN >= 0) digitalWrite(EN_PIN, on ? LOW : HIGH);
#else
  if (on) { motorX.enableOutputs(); motorY.enableOutputs(); }
  else    { motorX.disableOutputs(); motorY.disableOutputs(); }
#endif
}

bool moving() { return motorX.distanceToGo() != 0 || motorY.distanceToGo() != 0; }

#if USE_LIMIT_SWITCHES
bool homeAxis(AccelStepper &m, int limitPin) {
  m.setMaxSpeed(HOME_SPEED);
  m.setSpeed(-HOME_SPEED);
  long n = 0;
  while (digitalRead(limitPin) == HIGH) {          // HIGH = not pressed (pull-up)
    if (m.runSpeed()) n++;
    if (n > HOME_MAX_STEPS) return false;
    if (Serial.available() && Serial.peek() == 'S') return false;
    yield();
  }
  m.setCurrentPosition(0);
  m.runToNewPosition(HOME_BACKOFF);                 // back off the switch
  m.setCurrentPosition(0);
  m.setMaxSpeed(MAX_SPEED);
  return true;
}
#endif

void handle(String cmd) {
  cmd.trim();
  if (cmd.length() == 0) return;
  char c = toupper(cmd[0]);

  if (c == 'G') {
    long x, y;
    if (sscanf(cmd.c_str() + 1, "%ld %ld", &x, &y) != 2) { Serial.println("error: use G <x> <y>"); return; }
    if (!motorsOn) setMotors(true);
    motorX.moveTo(x);
    motorY.moveTo(y);
    Serial.println("ok");
  } else if (c == 'Z') {
    motorX.setCurrentPosition(0);
    motorY.setCurrentPosition(0);
    Serial.println("ok");
  } else if (c == 'H') {
#if USE_LIMIT_SWITCHES
    if (!motorsOn) setMotors(true);
    bool ok = homeAxis(motorX, X_LIMIT) && homeAxis(motorY, Y_LIMIT);
    Serial.println(ok ? "ok" : "error: homing failed (switch not reached or stopped)");
#else
    Serial.println("error: no limit switches (USE_LIMIT_SWITCHES 0). Put the charger at home and send Z");
#endif
  } else if (c == 'S') {
    motorX.stop();
    motorY.stop();
    Serial.println("ok");
  } else if (c == '?') {
    Serial.printf("pos %ld %ld moving %d\n", motorX.currentPosition(), motorY.currentPosition(), moving() ? 1 : 0);
  } else if (c == 'E') {
    int on = 1;
    sscanf(cmd.c_str() + 1, "%d", &on);
    setMotors(on != 0);
    Serial.println("ok");
  } else {
    Serial.println("error: unknown command");
  }
}

void setup() {
  Serial.begin(115200);
#if MOTOR_TYPE == 1
  if (EN_PIN >= 0) pinMode(EN_PIN, OUTPUT);
#endif
#if USE_LIMIT_SWITCHES
  pinMode(X_LIMIT, INPUT_PULLUP);
  pinMode(Y_LIMIT, INPUT_PULLUP);
#endif
  motorX.setMaxSpeed(MAX_SPEED);
  motorX.setAcceleration(ACCEL);
  motorY.setMaxSpeed(MAX_SPEED);
  motorY.setAcceleration(ACCEL);
  setMotors(true);
  Serial.println("ready charger_mover");
}

unsigned long idleSince = 0;

void loop() {
  while (Serial.available()) {
    char ch = Serial.read();
    if (ch == '\n' || ch == '\r') { handle(line); line = ""; }
    else if (line.length() < 64) line += ch;
  }
  motorX.run();
  motorY.run();

#if MOTOR_TYPE == 2
  // 28BYJ-48 coils get hot if left powered; release them after 2 s idle (they hold by gearing).
  if (moving()) idleSince = millis();
  else if (motorsOn && millis() - idleSince > 2000) setMotors(false);
#endif
}
