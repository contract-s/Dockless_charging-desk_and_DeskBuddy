/*
  step_test.ino  —  bare-minimum motor test. No libraries, no commands.
  Turns the motor slowly one way for 2 s, waits, turns it back, forever.
  The blue LED on the ESP32 (GPIO 2) flips each time the direction changes.

  If THIS turns the motor, the wiring is right and the problem was in charger_mover.
  If this does NOT turn it either, the problem is wiring (STEP pin, motor coil pairs).

  Wiring (same as charger_mover): STEP -> GPIO 26, DIR -> GPIO 27, EN -> GPIO 25.
  Upload, then open Serial Monitor at 115200 to see what it's doing.
  When done, upload charger_mover.ino again.
*/

const int STEP_PIN = 26, DIR_PIN = 27, EN_PIN = 25, LED_PIN = 2;
const int STEPS = 800;          // with TMC2209 at 1/8 microstepping = half a turn
const int HALF_PERIOD_US = 1000; // 500 steps/s: slow and easy for the motor

void setup() {
  Serial.begin(115200);
  pinMode(STEP_PIN, OUTPUT);
  pinMode(DIR_PIN, OUTPUT);
  pinMode(EN_PIN, OUTPUT);
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(EN_PIN, LOW);     // LOW = driver on
  Serial.println("step_test: motor should turn back and forth");
}

void loop() {
  static bool dir = false;
  dir = !dir;
  digitalWrite(DIR_PIN, dir ? HIGH : LOW);
  digitalWrite(LED_PIN, dir ? HIGH : LOW);
  Serial.printf("turning %s (%d steps)\n", dir ? "forward" : "back", STEPS);
  for (int i = 0; i < STEPS; i++) {
    digitalWrite(STEP_PIN, HIGH);
    delayMicroseconds(HALF_PERIOD_US);
    digitalWrite(STEP_PIN, LOW);
    delayMicroseconds(HALF_PERIOD_US);
  }
  delay(1000);
}
