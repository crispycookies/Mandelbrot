//       $Id: pfc_threading.h 39738 2019-11-27 12:38:45Z p20068 $
//      $URL: https://svn01.fh-hagenberg.at/bin/cepheiden/vocational/teaching/SE-Master/MPV3/Inhalt/source/common-2/threading/src/pfc_threading.h $
// $Revision: 39738 $
//     $Date: 2019-11-27 13:38:45 +0100 (Mi., 27 Nov 2019) $
//   Creator: Peter Kulczycki
//  Creation: October, 2019
//   $Author: p20068 $
// Copyright: (c) 2019 Peter Kulczycki (peter.kulczycki<AT>fh-hagenberg.at)
//   License: This document contains proprietary information belonging to
//            University of Applied Sciences Upper Austria, Campus Hagenberg. It
//            is distributed under the Boost Software License, Version 1.0 (see
//            http://www.boost.org/LICENSE_1_0.txt).

#pragma once

#include <future>
#include <thread>
#include <vector>

// -------------------------------------------------------------------------------------------------

namespace pfc {

    inline auto hardware_concurrency () noexcept {
        return std::max <decltype (std::thread::hardware_concurrency ())> (1, std::thread::hardware_concurrency ());
    }

    template <typename size_t> constexpr auto load_per_task (size_t const task, size_t const tasks, size_t const size) noexcept {
        return size / tasks + ((task < (size % tasks)) ? size_t {1} : size_t {0});
    }

}   // namespace pfc

// -------------------------------------------------------------------------------------------------

namespace pfc {

    class task_group final {
    public:
        explicit task_group () = default;

        task_group (task_group const &) = delete;
        task_group (task_group &&) = default;

        ~task_group () {
            join_all ();
        }

        task_group & operator = (task_group const &) = delete;
        task_group & operator = (task_group &&) = default;

        template <typename fun_t, typename ...args_t> void add (fun_t && fun, args_t && ...args) {
            m_group.push_back (
                    std::async (std::launch::async, std::forward <fun_t> (fun), std::forward <args_t> (args)...)
            );
        }

        void join_all () {
            for (auto & f : m_group) f.wait ();
        }

    private:
        std::vector <std::future <void>> m_group;
    };

}   // namespace pfc

// -------------------------------------------------------------------------------------------------

namespace pfc {

    class thread_group final {
    public:
        explicit thread_group () = default;

        thread_group (thread_group const &) = delete;
        thread_group (thread_group &&) = default;

        ~thread_group () {
            join_all ();
        }

        thread_group & operator = (thread_group const &) = delete;
        thread_group & operator = (thread_group &&) = default;

        template <typename fun_t, typename ...args_t> void add (fun_t && fun, args_t && ...args) {
            m_group.emplace_back (std::forward <fun_t> (fun), std::forward <args_t> (args)...);
        }

        void join_all () {
            for (auto & t : m_group) if (t.joinable ()) t.join ();
        }

    private:
        std::vector <std::thread> m_group;
    };

}   // namespace pfc

// -------------------------------------------------------------------------------------------------

namespace pfc {

    template <typename size_t, typename fun_t> void parallel_range (task_group & group, size_t const tasks, size_t const size, fun_t && fun) {
        size_t begin {0};
        size_t end   {0};

        for (size_t t {0}; t < tasks; ++t) {
            end += load_per_task (t, tasks, size);

            if (end > begin) {
                group.add (std::forward <fun_t> (fun), t, begin, end);
            }

            begin = end;
        }
    }

    template <typename size_t, typename fun_t> void parallel_range (size_t const tasks, size_t const size, fun_t && fun) {
        task_group group; pfc::parallel_range (group, tasks, size, std::forward <fun_t> (fun));
    }

}   // namespace pfc