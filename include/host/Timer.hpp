#pragma once

#include <chrono>
#include <iomanip>
#include <ostream>

class Timer{
    public:
        Timer(){ m_tick = std::chrono::high_resolution_clock::now(); }
        std::chrono::nanoseconds Elapsed(){ return std::chrono::high_resolution_clock::now() - m_tick; }
    private:
        std::chrono::time_point<std::chrono::high_resolution_clock> m_tick;
};

// Beautify output for std::chrono::nanoseconds
// Source[Modified]: https://stackoverflow.com/questions/22590821/convert-stdduration-to-human-readable-time
std::ostream& operator<<(std::ostream& os, std::chrono::nanoseconds ns){
    using namespace std::chrono;
    auto h = duration_cast<hours>(ns);
    ns -= h;
    auto m = duration_cast<minutes>(ns);
    ns -= m;
    auto s = duration_cast<seconds>(ns);
    ns -= s;
    auto ms = duration_cast<milliseconds>(ns);
    ns -= ms;

    char fill = os.fill('0');
    if (h.count())
        os << h.count() << "h ";
    if (h.count() || m.count())
        os << std::setw(h.count() ? 2 : 1) << m.count() << "m ";
    if (h.count() || m.count() || s.count())
        os << std::setw(h.count() || m.count() ? 2 : 1) << s.count() << "s ";
    if (h.count() || m.count() || s.count() || ms.count())
        os << std::setw(h.count() || m.count() || s.count() ? 3 : 1) << ms.count() << "ms";

    os.fill(fill);
    return os;
}