#include <sys/time.h>
#include <ctime>

class StopWatchInterface {
  public:
    StopWatchInterface() {}
    virtual ~StopWatchInterface() {}

  public: 
    virtual void start() = 0;
    virtual void stop() = 0;
    virtual void reset() = 0;
    virtual float getTime() = 0;
    virtual float getAverageTime() = 0;
};

class StopWatchLinux : public StopWatchInterface {
  public:
  StopWatchLinux() 
    : start_time(),
      diff_time(0.0),
      total_time(0.0),
      running(false),
      clock_sessions(0) {}
  virtual ~StopWatchLinux() {}

  public:
   inline void start();
   inline void stop();
   inline void reset();
   inline float getTime();
   inline float getAverageTime();
  private:
   inline float getDiffTime();

  private:
   struct timeval start_time;
   float diff_time;
   float total_time;
   bool running;
   int clock_sessions;
};

inline void StopWatchLinux::start() {
  gettimeofday(&start_time, 0);
}

inline void StopWatchLinux::stop() {
  diff_time = getDiffTime();
  total_time += diff_time;
  running = false;
  clock_sessions++;
}

inline void StopWatchLinux::reset() {
  diff_time = 0;
  total_time = 0;
  clock_sessions = 0;
  if (running) {
    gettimeofday(&start_time, 0);
  }
}

inline float StopWatchLinux::getTime(){
  float retval = total_time;
  if (running) {
    retval += getDiffTime();
  }
  return retval;
}

inline float StopWatchLinux::getAverageTime() {
  return (clock_sessions > 0) ? (total_time/clock_sessions) : 0.0f; 
}

inline float StopWatchLinux::getDiffTime() {
  struct timeval t_time;
  gettimeofday(&t_time, 0);
  return static_cast<float>(1000.0 * (t_time.tv_sec - start_time.tv_sec) +
                            (0.001 * (t_time.tv_usec - start_time.tv_usec)));
}

inline bool sdkCreateTimer(StopWatchInterface **timer_interface) {
  *timer_interface = reinterpret_cast<StopWatchInterface *>(new StopWatchLinux());
  return (*timer_interface != NULL) ? true : false;
}

inline bool sdkDeleteTimer(StopWatchInterface **timer_interface) {
  if (*timer_interface) {
    delete *timer_interface;
    *timer_interface = NULL;
  }
  return true;
}

inline bool sdkStartTimer(StopWatchInterface **timer_interface) {
  if (*timer_interface) {
    (*timer_interface)->start();
  }
  return true;
}
inline bool sdkResetTimer(StopWatchInterface **timer_interface) {
  if (*timer_interface) {
    (*timer_interface)->reset();
  }
  return true;
}
inline float sdkGetTimerValue(StopWatchInterface **timer_interface) {
  if (*timer_interface) {
    return (*timer_interface)->getTime();
  } else {
    return 0.0f;
  }

}
