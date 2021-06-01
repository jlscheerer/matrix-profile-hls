#pragma once

#include <iostream>
#include <queue>
#include <stdexcept>

namespace Mock {

    static bool all_streams_empty{true};
    static bool read_from_empty_stream{false};

    static void Reset(){
        all_streams_empty = true;
        read_from_empty_stream = false;
    }

    template<typename T, size_t depth>
    class stream {
        private:
            std::queue<T> m_queue;
        public:
            void write(const T element) {
                m_queue.push(element);
            }
            T read(){
                if(!m_queue.size()){
                    read_from_empty_stream = true;
                    return T();
                }
                T result{m_queue.front()};
                m_queue.pop();
                return result;
            }
            ~stream(){ all_streams_empty &= m_queue.size() == 0; }
    };

};