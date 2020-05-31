//       $Id: pfc_timing.h 39478 2019-10-10 07:20:36Z p20068 $
//      $URL: https://svn01.fh-hagenberg.at/bin/cepheiden/vocational/teaching/SE-Master/MPV3/Inhalt/source/common-2/timing/src/pfc_timing.h $
// $Revision: 39478 $
//     $Date: 2019-10-10 09:20:36 +0200 (Do., 10 Okt 2019) $
//   Creator: Peter Kulczycki
//  Creation: October, 2019
//   $Author: p20068 $
// Copyright: (c) 2019 Peter Kulczycki (peter.kulczycki<AT>fh-hagenberg.at)
//   License: This document contains proprietary information belonging to
//            University of Applied Sciences Upper Austria, Campus Hagenberg. It
//            is distributed under the Boost Software License, Version 1.0 (see
//            http://www.boost.org/LICENSE_1_0.txt).

#pragma once

#include "pfc_threading.h"

#include <chrono>
#include <functional>

// -------------------------------------------------------------------------------------------------

namespace pfc {

    using default_clock_t = std::chrono::high_resolution_clock;

}   // namespace pfc

// -------------------------------------------------------------------------------------------------

namespace pfc {

    template <typename D> constexpr char const * time_unit () noexcept {
        using period_t = typename D::period;

        if constexpr (std::is_same_v <period_t, std::nano>)
            return "ns";

        else if constexpr (std::is_same_v <period_t, std::micro>)
            return "us";

        else if constexpr (std::is_same_v <period_t, std::milli>)
            return "ms";

        else if constexpr (std::is_same_v <period_t, std::ratio <1, 1>>)
            return "s";

        else if constexpr (std::is_same_v <period_t, std::ratio <1, 60>>)
            return "min";

        else if constexpr (std::is_same_v <period_t, std::ratio <1, 60 * 60>>)
            return "h";

        else
            return "";
    }

    template <typename D> constexpr auto time_unit (D const &) noexcept {
        return time_unit <D> ();
    }

}   // namespace pfc

// -------------------------------------------------------------------------------------------------

namespace pfc {

    template <typename C = default_clock_t, typename S, typename F, typename ...A> auto timed_run (S const n, F && fun, A && ...args) noexcept (std::is_nothrow_invocable_v <F, A...>) {
        //static_assert (C::is_steady);
        static_assert (std::is_integral_v <S>);

        using duration_t = typename C::duration;

        duration_t elapsed {};

        if (0 < n) {
            auto const start {C::now ()};

            for (int i {0}; i < n; ++i) {
                std::invoke (std::forward <F> (fun), std::forward <A> (args)...);
            }

            elapsed = (C::now () - start) / n;
        }

        return elapsed;
    }

    template <typename C = default_clock_t, typename F, typename ...A> auto timed_run (F && fun, A && ...args) noexcept (std::is_nothrow_invocable_v <F, A...>) {
        return timed_run (1, std::forward <F> (fun), std::forward <A> (args)...);
    }

}   // namespace pfc

// -------------------------------------------------------------------------------------------------

namespace pfc {

    template <typename C = default_clock_t, typename D = std::chrono::seconds> void warm_up_cpu (D const how_long = D {5}) {
        //static_assert (C::is_steady);

        auto         cores {hardware_concurrency ()};
        thread_group group {};

        while (0 < cores--) {
            group.add ([how_long, start = C::now ()] {
                while ((C::now () - start) < how_long);
            });
        }
    }

}   // namespace pfc

// -------------------------------------------------------------------------------------------------

template <typename ...T> auto & operator << (std::ostream & lhs, std::chrono::duration <T...> const & rhs) {
    return lhs << rhs.count () << pfc::time_unit (rhs);
}


