//       $Id: pfc_random.h 39741 2019-11-28 09:09:45Z p20068 $
//      $URL: https://svn01.fh-hagenberg.at/bin/cepheiden/vocational/teaching/MBI/ADS1/2018-WS/VL/src/random/src/pfc_random.h $
// $Revision: 39741 $
//     $Date: 2019-11-28 10:09:45 +0100 (Do., 28 Nov 2019) $
//   Creator: Peter Kulczycki
//  Creation: October, 2019
//   $Author: p20068 $
// Copyright: (c) 2019 Peter Kulczycki (peter.kulczycki<AT>fh-hagenberg.at)
//   License: This document contains proprietary information belonging to
//            University of Applied Sciences Upper Austria, Campus Hagenberg. It
//            is distributed under the Boost Software License, Version 1.0 (see
//            http://www.boost.org/LICENSE_1_0.txt).

#pragma once

#include <algorithm>
#include <random>
#include <type_traits>
#include <vector>

// -------------------------------------------------------------------------------------------------

namespace pfc::util {

    template <typename E> constexpr bool is_std_random_engine_v{
       std::is_same_v <E, std::knuth_b> ||
       std::is_same_v <E, std::minstd_rand> ||
       std::is_same_v <E, std::minstd_rand0> ||
       std::is_same_v <E, std::mt19937> ||
       std::is_same_v <E, std::mt19937_64> ||
       std::is_same_v <E, std::ranlux24> ||
       std::is_same_v <E, std::ranlux24_base> ||
       std::is_same_v <E, std::ranlux48> ||
       std::is_same_v <E, std::ranlux48_base>
    };

    template <typename E> struct std_random_engine_traits final {
        static_assert (
            is_std_random_engine_v <E>,
            "pfc::util::std_random_engine_traits<E>: E must be a random engine from namespace std"
            );

        constexpr static std::size_t state_size{ 1 };
    };

    template <> struct std_random_engine_traits <std::mt19937> final {
        constexpr static std::size_t state_size{ std::mt19937::state_size };
    };

    template <> struct std_random_engine_traits <std::mt19937_64> final {
        constexpr static std::size_t state_size{ std::mt19937_64::state_size };
    };

}   // namespace pfc::util

// -------------------------------------------------------------------------------------------------

namespace pfc::util {

    template <typename E> auto make_seeds() {
        std::vector <std::seed_seq::result_type> seeds(std_random_engine_traits <E>::state_size);

        std::generate(
            std::begin(seeds),
            std::end(seeds),
            std::minstd_rand{ std::random_device {} () }
        );

        return seeds;
    }

    template <typename E> auto make_random_engine() {
        auto const    seeds{ make_seeds <E>() };
        std::seed_seq sequence(std::begin(seeds), std::end(seeds));

        return E{ sequence };
    }

}   // namespace pfc::util

// -------------------------------------------------------------------------------------------------

namespace pfc {

    using default_random_engine = std::mt19937_64;

    /**
     * If T is an integral type, get_random_uniform produces random integer values x
     * uniformly distributed on the interval [l,u] according to the probability
     * density function P(x) = 1 / (b − a + 1).
     *
     * If T is a floating-point type, get_random_uniform produces random
     * floating-point values x uniformly distributed on the interval [l,u[
     * according to the probability density function P(x) = 1 / (b − a).
     *
     * For more information see https://en.cppreference.com/w/cpp/numeric/random
     */
    template <typename T, typename E = default_random_engine> auto get_random_uniform(T const l, T const u) {
        static_assert (util::is_std_random_engine_v <E>, "pfc::get_random_uniform<T,E>: E must be a random engine from namespace std");
        static_assert (std::is_arithmetic_v <T>, "pfc::get_random_uniform<T,E>: T must be an arithmetic type");

        static auto engine{ util::make_random_engine <E>() };

        if constexpr (std::is_integral_v <T>) {
            return std::uniform_int_distribution  <T> {l, u} (engine);
        }
        else {
            return std::uniform_real_distribution <T> {l, u} (engine);
        }
    }

}   // namespace pfc