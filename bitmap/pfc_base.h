//        $Id: pfc_base.h 40278 2020-02-28 18:46:06Z p20068 $
//       $URL: https://svn01.fh-hagenberg.at/bin/cepheiden/vocational/teaching/SE-Master/MPV3/2018-WS/ILV/src/bitmap/src/pfc_base.h $
//  $Revision: 40278 $
//      $Date: 2020-02-28 19:46:06 +0100 (Fr., 28 Feb 2020) $
//    $Author: p20068 $
//
//    Creator: Peter Kulczycki (peter.kulczycki<AT>fh-hagenberg.at)
//   Creation:
//  Copyright: (c) 2020 Peter Kulczycki (peter.kulczycki<AT>fh-hagenberg.at)
//
//    License: This document contains proprietary information belonging to
//             University of Applied Sciences Upper Austria, Campus
//             Hagenberg. It is distributed under the Boost Software License,
//             Version 1.0 (see http://www.boost.org/LICENSE_1_0.txt).
//
// Annotation: This file is part of the code snippets handed out during one
//             of my HPC lessons held at the University of Applied Sciences
//             Upper Austria, Campus Hagenberg.

#pragma once

#include "./pfc_libraries.h"

#include "./pfc_traits.h"
#include "./pfc_types.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <deque>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace pfc {

// -------------------------------------------------------------------------------------------------

#if defined PFC_HAVE_GSL

template <typename T> constexpr void memcpy (gsl::span <T> & dst, gsl::span <T> const & src) noexcept {
   std::memcpy (std::data (dst), std::data (src), std::min (dst.size_bytes (), src.size_bytes ()));
}

template <typename T> constexpr void memset (gsl::span <T> & span, int const byte) noexcept {
   std::memset (std::data (span), byte, span.size_bytes ());
}

#endif   // PFC_HAVE_GSL

template <typename T> constexpr void memset (T & obj, int const byte) noexcept {
   std::memset (&obj, byte, sizeof (T));
}

// -------------------------------------------------------------------------------------------------

template <typename T> auto & read (std::istream & in, T & obj) noexcept {
   if (in) {
      in.read (reinterpret_cast <std::remove_reference_t <decltype (in)>::char_type *> (&obj), sizeof (T));
   }

   return in;
}

template <typename T> auto & write (std::ostream & out, T const & obj) noexcept {
   if (out) {
      out.write (reinterpret_cast <std::remove_reference_t <decltype (out)>::char_type *> (const_cast <T *> (&obj)), sizeof (T));
   }

   return out;
}

#if defined PFC_HAVE_GSL

template <typename T> auto & read (std::istream & in, gsl::span <T> & span) noexcept {
   if (in) {
      in.read (reinterpret_cast <std::remove_reference_t <decltype (in)>::char_type *> (std::data (span)), span.size_bytes ());
   }

   return in;
}

template <typename T> auto & write (std::ostream & out, gsl::span <T> const & span) noexcept {
   if (out) {
      out.write (reinterpret_cast <std::remove_reference_t <decltype (out)>::char_type *> (std::data (span)), span.size_bytes ());
   }

   return out;
}

#endif   // PFC_HAVE_GSL

// -------------------------------------------------------------------------------------------------

template <template <typename ...> typename cont_t, typename ...args_t> auto make_from_values (args_t && ...args) {
   using common_t = std::common_type_t <args_t...>; return cont_t <common_t> {common_t (std::forward <args_t> (args))...};
}

// -------------------------------------------------------------------------------------------------

class scoped_ostream_redirector final {
   static auto & null () {
      static std::ostream empty {nullptr}; return empty;
   }

   public:
      scoped_ostream_redirector () = default;

      explicit scoped_ostream_redirector (std::ostream & out)
         : m_p_out  {&out}
         , m_buffer {out.rdbuf (null ().rdbuf ())} {
      }

      explicit scoped_ostream_redirector (std::ostream & out, std::ostream & to)
         : m_p_out  {&out}
         , m_buffer {out.rdbuf (to.rdbuf ())} {
      }

      scoped_ostream_redirector (scoped_ostream_redirector const &) = delete;
      scoped_ostream_redirector (scoped_ostream_redirector &&) = default;

     ~scoped_ostream_redirector () {
         if (m_p_out) {
            m_p_out->rdbuf (m_buffer);
         }
      }

      scoped_ostream_redirector & operator = (scoped_ostream_redirector const &) = delete;
      scoped_ostream_redirector & operator = (scoped_ostream_redirector &&) = delete;

   private:
      std::ostream *                                     m_p_out  {nullptr};   // non owning
      decltype (std::declval <std::ostream> ().rdbuf ()) m_buffer {};          //
};

template <typename fun_t> void invoke_redirected (fun_t && fun) {
   pfc::scoped_ostream_redirector const redirect; std::invoke (std::forward <fun_t> (fun));
}

template <typename fun_t> void invoke_redirected (std::ostream & out, fun_t && fun) {
   pfc::scoped_ostream_redirector const redirect {out}; std::invoke (std::forward <fun_t> (fun));
}

template <typename fun_t> void invoke_redirected (std::ostream & out, std::ostream & to, fun_t && fun) {
   pfc::scoped_ostream_redirector const redirect {out, to}; std::invoke (std::forward <fun_t> (fun));
}

template <typename fun_t> void invoke_redirected (std::ostream & out, std::ostream && to, fun_t && fun) {
   pfc::scoped_ostream_redirector const redirect {out, to}; std::invoke (std::forward <fun_t> (fun));
}

// -------------------------------------------------------------------------------------------------

template <typename itor_t> std::ostream & print_range (itor_t const b, itor_t const e, std::ostream & out) {
   out << '{'; auto first {true};

   std::for_each (b, e, [&first, &out] (auto const & e) {
      if (!first) {
         out << ',';
      }

      out << e; first = false;
   });

   return out << '}';
}

// -------------------------------------------------------------------------------------------------

template <typename T1, typename T2> constexpr auto ceil_div (T1 const a, T2 const b) noexcept {
   static_assert (pfc::is_integral_v <T1>, "");   // !pwk: backward compatibility (e.g. for nvcc)
   static_assert (pfc::is_integral_v <T2>, "");   // !pwk: backward compatibility (e.g. for nvcc)

   auto const m {((a < 0) ? -a : a) % b};

   if (m == 0) {
      return a / b;

   } else if (a < 0) {
      return (a + m) / b;

   } else {
      return (a + b - m) / b;
   }
}

template <typename T1, typename T2> constexpr auto ceil_div_2 (T1 const a, T2 const b) noexcept {
   static_assert (pfc::is_integral_v <T1>, "");   // !pwk: backward compatibility (e.g. for nvcc)
   static_assert (pfc::is_integral_v <T2>, "");   // !pwk: backward compatibility (e.g. for nvcc)

   if /*constexpr*/ (pfc::is_integral_unsigned_v <T1>) {   // !pwk: backward compatibility (e.g. for nvcc)
      return             (b > 0) ? (a + b - 1) / b : 0;
   } else {
      return (a >= 0) && (b > 0) ? (a + b - 1) / b : 0;
   }
}

template <typename value_t> constexpr auto clamp (value_t const & value, value_t const & left, value_t const & right) noexcept {
   return std::max (left, std::min (right, value));
}

template <typename value_t> constexpr auto clamp_indirect (double const f, value_t const & left, value_t const & right) {
   return static_cast <value_t> (left + (right - left) * pfc::clamp (f, 0.0, 1.0));   // !pwk: use std::clamp
}

constexpr int digits (unsigned long long i) noexcept {
   int n {1};

   if (i >= 100'000'000) { n += 8; i /= 100'000'000; }
   if (i >=      10'000) { n += 4; i /=      10'000; }
   if (i >=         100) { n += 2; i /=         100; }
   if (i >=          10) { n += 1;                   }

   return n;
}

template <typename T> constexpr bool is_even (T const v) noexcept {
   static_assert (pfc::is_integral_v <T>, ""); return (v % 2) == 0;   // !pwk : backward compatibility (e.g. for nvcc)
}

template <typename T> constexpr bool is_odd (T const v) noexcept {
   return !pfc::is_even (v);
}

template <typename T> CATTR_GPU_ENABLED_INLINE constexpr auto isqrt (T const x) noexcept {
   static_assert (pfc::is_integral_v <T>, ""); using value_t = std::make_unsigned_t <T>;   // !pwk : backward compatibility (e.g. for nvcc)

   constexpr auto a {sizeof (value_t) * 4};
   constexpr auto b {sizeof (value_t) * 8 - 2};

   value_t m {};
   value_t n {};
   value_t X {static_cast <value_t> (x)};

   for (int i {0}; i < a; ++i) {
      m = (m << 2) + (X >> b); ++(n <<= 1); X <<= 2; n <= m ? m -= n++ : --n;
   }

   return n >> 1;
}

template <typename T> constexpr bool is_square (T const x) noexcept {
   static_assert (pfc::is_integral_v <T>, ""); auto const r {pfc::isqrt (x)}; return r * r == x;   // !pwk : backward compatibility (e.g. for nvcc)
}

template <typename T> constexpr int size (T const & t) noexcept {
   return static_cast <int> (std::size (t));
}

template <typename T> constexpr T square (T const & x) noexcept {
   return x * x;
}

constexpr int digital_root (int const x) noexcept {   // https://de.wikipedia.org/wiki/Quersumme#Einstellige_(oder_iterierte)_Quersumme
   if (x <= 0) {
      return 0;
   }

   int const m {x % 9}; return (m == 0) ? 9 : m;
}

template <typename T, typename U> constexpr bool divides (T const a, U const b) noexcept {
   static_assert (pfc::is_integral_v <T>, "");   // !pwk : backward compatibility (e.g. for nvcc)
   static_assert (pfc::is_integral_v <U>, "");   // !pwk : backward compatibility (e.g. for nvcc)

   return (a == 0) ? false : (b % a) == 0;
}

// -------------------------------------------------------------------------------------------------

template <typename ratio_t, typename value_t> constexpr auto prefix_cast (value_t const & value) noexcept {
   return static_cast <double> (value) * ratio_t::num / ratio_t::den;
}

template <typename enum_t> constexpr auto underlying_cast (enum_t const & value) noexcept {
   return static_cast <std::underlying_type_t <enum_t>> (value);
}

// -------------------------------------------------------------------------------------------------

using hectonano = std::ratio <1, std::nano::den / 100>;   // 100 nanos

template <typename ratio_t> constexpr char const * unit_prefix () noexcept {
   static_assert (pfc::is_ratio_v <ratio_t>, "");   // !pwk: backward compatibility (e.g. for nvcc)

   if (std::is_same_v <ratio_t, std::atto>)
      return "a";

   else if (std::is_same_v <ratio_t, std::femto>)
      return "f";

   else if (std::is_same_v <ratio_t, std::pico>)
      return "p";

   else if (std::is_same_v <ratio_t, std::nano>)
      return "n";

   else if (std::is_same_v <ratio_t, pfc::hectonano>)
      return "hn";

   else if (std::is_same_v <ratio_t, std::micro>)
      return "u";

   else if (std::is_same_v <ratio_t, std::milli>)
      return "m";

   else if (std::is_same_v <ratio_t, std::centi>)
      return "c";

   else if (std::is_same_v <ratio_t, std::deci>)
      return "d";

   else if (std::is_same_v <ratio_t, std::deca>)
      return "da";

   else if (std::is_same_v <ratio_t, std::hecto>)
      return "h";

   else if (std::is_same_v <ratio_t, std::kilo>)
      return "k";

   else if (std::is_same_v <ratio_t, std::mega>)
      return "M";

   else if (std::is_same_v <ratio_t, std::giga>)
      return "G";

   else if (std::is_same_v <ratio_t, std::tera>)
      return "T";

   else if (std::is_same_v <ratio_t, std::peta>)
      return "P";

   else if (std::is_same_v <ratio_t, std::exa>)
      return "E";

   else
      return "?";
}

// -------------------------------------------------------------------------------------------------

template <typename T> class range final {
   using itor_t = T;

   public:
      constexpr explicit range (itor_t const & b, itor_t const & e) : m_begin {b}, m_end {e} {
      }

      constexpr auto begin () const {
         return m_begin;
      }

      constexpr auto end () const {
         return m_end;
      }

   private:
      itor_t m_begin {};
      itor_t m_end   {};
};

template <typename itor_t> constexpr auto make_range (itor_t const & b, itor_t const & e) {
   return pfc::range <itor_t> {b, e};
}

template <typename value_t> constexpr auto make_range (value_t * const b, int const s) {
   return pfc::range <value_t *> {b, b + s};
}

template <typename value_t, int s> constexpr auto make_range (value_t (& b) [s]) {
   return pfc::range <value_t *> {b, b + s};
}

template <typename itor_t> constexpr auto make_range (std::pair <itor_t, itor_t> const & range) {
   return pfc::range <itor_t> {std::get <0> (range), std::get <1> (range)};
}

template <typename itor_t> constexpr auto make_range (std::tuple <itor_t, itor_t> const & range) {
   return pfc::range <itor_t> {std::get <0> (range), std::get <1> (range)};
}

template <typename value_t> constexpr auto make_range (std::istream & in) {
   return pfc::make_range (std::istream_iterator <value_t> {in}, {});
}

// -------------------------------------------------------------------------------------------------

}   // namespace pfc

template <typename T, int n> std::ostream & operator << (std::ostream & lhs, std::array <T, n> const & rhs) {
   return pfc::print_range (std::begin (rhs), std::end (rhs), lhs);
}

template <typename ...T> std::ostream & operator << (std::ostream & lhs, std::deque <T...> const & rhs) {
   return pfc::print_range (std::begin (rhs), std::end (rhs), lhs);
}

template <typename ...T> std::ostream & operator << (std::ostream & lhs, std::vector <T...> const & rhs) {
   return pfc::print_range (std::begin (rhs), std::end (rhs), lhs);
}
