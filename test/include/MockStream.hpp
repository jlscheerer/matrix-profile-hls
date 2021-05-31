#pragma once

#include <iostream>
#include <queue>

namespace mock {

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
                    // TODO: Throw Error here!
                    std::cout << "Trying to read from empty stream!" << std::endl;
                    return T();
                }
                T result{m_queue.front()};
                m_queue.pop();
                return result;
            }
    };

};