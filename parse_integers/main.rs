use std::{mem, mem::MaybeUninit, thread};

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

const SPACE: u8 = b'\n';

fn parse_naive_loop(s: &[u8]) -> u64 {
    let mut sum = 0;

    let mut cur = 0;
    for &c in s {
        if c >= b'0' && c <= b'9' {
            cur = cur * 10 + (c - b'0') as u64;
        } else {
            sum += cur;
            cur = 0;
        }
    }
    sum + cur
}

mod simd256 {
    use super::*;

    const MAX_NUMBER_LEN: usize = 10;
    pub type ShuffleMasks =
        [[[(__m256i, __m256i); MAX_NUMBER_LEN]; MAX_NUMBER_LEN]; MAX_NUMBER_LEN];

    pub unsafe fn from_slice(s: &[u8]) -> __m256i {
        _mm256_lddqu_si256(s.as_ptr() as *const __m256i)
    }

    pub unsafe fn gen_shuffle_masks() -> ShuffleMasks {
        unsafe fn get_mask(d1: usize, d2: usize, d3: usize) -> (__m256i, __m256i) {
            fn fill_mask(
                m: &mut [u8],
                d: usize,
                src_offset: usize,
                dst_pos: usize,
                dst_top_pos: usize,
            ) {
                if d > 8 {
                    let d = d - 8;
                    let z = 2 - d;
                    for i in 0..8 {
                        m[i + dst_pos] = (i + d + src_offset) as u8;
                    }
                    for i in 0..d {
                        m[dst_top_pos + z + i] = (i + src_offset) as u8;
                    }
                } else {
                    let z = 8 - d;
                    for i in 0..d {
                        m[i + z + dst_pos] = (i + src_offset) as u8;
                    }
                }
            }

            let mut m = [d1 as u8; 32]; // string should contain space character at d1 index
            fill_mask(&mut m, d1, 0, 0, 26);
            fill_mask(&mut m, d2, d1 + 1, 8, 28);
            fill_mask(&mut m, d3, d1 + d2 + 2, 16, 30);

            let m = from_slice(&m);

            // prepare mask for using in shuffle
            let (k0, k1) = {
                let a0 = 0x70u8 as i8;
                let a1 = 0xf0u8 as i8;
                let k0 = _mm256_setr_epi8(
                    a0, a0, a0, a0, a0, a0, a0, a0, a0, a0, a0, a0, a0, a0, a0, a0, a1, a1, a1, a1,
                    a1, a1, a1, a1, a1, a1, a1, a1, a1, a1, a1, a1,
                );
                let k1 = _mm256_setr_epi8(
                    a1, a1, a1, a1, a1, a1, a1, a1, a1, a1, a1, a1, a1, a1, a1, a1, a0, a0, a0, a0,
                    a0, a0, a0, a0, a0, a0, a0, a0, a0, a0, a0, a0,
                );
                (k0, k1)
            };
            (_mm256_add_epi8(m, k0), _mm256_add_epi8(m, k1))
        }

        let mut res = [[[(_mm256_setzero_si256(), _mm256_setzero_si256()); MAX_NUMBER_LEN];
            MAX_NUMBER_LEN]; MAX_NUMBER_LEN];

        for i in 0..MAX_NUMBER_LEN {
            for j in 0..MAX_NUMBER_LEN {
                for k in 0..MAX_NUMBER_LEN {
                    res[i][j][k] = get_mask(i + 1, j + 1, k + 1);
                }
            }
        }
        res
    }

    unsafe fn shuffle(x: __m256i, (shuffle0, shuffle1): (__m256i, __m256i)) -> __m256i {
        _mm256_or_si256(
            _mm256_shuffle_epi8(x, shuffle0),
            _mm256_shuffle_epi8(_mm256_permute4x64_epi64(x, 0b01_00_11_10), shuffle1),
        )
    }

    unsafe fn char_to_digit(x: __m256i) -> __m256i {
        let zero = _mm256_set1_epi8(b'0' as i8);
        _mm256_subs_epu8(x, zero)
    }

    unsafe fn find_next_nums(x: __m256i) -> (usize, usize, usize) {
        let separator = _mm256_set1_epi8(SPACE as i8);
        let separator_mask = _mm256_movemask_epi8(_mm256_cmpeq_epi8(x, separator));

        let d1 = separator_mask.trailing_zeros() as usize;
        let separator_mask = (separator_mask - 1) & separator_mask;
        let d2 = separator_mask.trailing_zeros() as usize - d1 - 1;
        let separator_mask = (separator_mask - 1) & separator_mask;
        let d3 = separator_mask.trailing_zeros() as usize - d1 - d2 - 2;
        debug_assert!(d1 <= MAX_NUMBER_LEN && d2 <= MAX_NUMBER_LEN && d3 <= MAX_NUMBER_LEN);

        (d1, d2, d3)
    }

    pub unsafe fn parse(mut s: &[u8], shuffle_masks: &ShuffleMasks) -> u64 {
        unsafe fn parse_impl(x: __m256i) -> u64 {
            let tens = _mm256_setr_epi8(
                10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 10, 1,
                10, 1, 10, 1, 10, 1, 10, 1,
            );
            let x = _mm256_maddubs_epi16(x, tens);

            let hundreds =
                _mm256_setr_epi16(100, 1, 100, 1, 100, 1, 100, 1, 100, 1, 100, 1, 1, 1, 1, 1);
            let x = _mm256_madd_epi16(x, hundreds);

            let mut v = [0_u32; 8];
            _mm256_storeu_si256(v.as_mut_ptr() as *mut __m256i, x);

            let v = v.map(|x| x as u64);

            let k = 10000;
            let a = v[0] * k + v[1];
            let b = v[2] * k + v[3];
            let c = v[4] * k + v[5];
            let d = v[6] + v[7];
            a + b + c + d * k * k
        }

        const MAX_GROUP_SIZE: usize = 28;

        let mut group = _mm256_setzero_si256();
        let mut group_size = 0;
        let mut res = 0;
        while s.len() > 32 {
            let x = from_slice(s);

            // find first three numbers
            let (d1, d2, d3) = find_next_nums(x);

            // shuffle digits for parsing
            let shuffle_mask = shuffle_masks[d1 - 1][d2 - 1][d3 - 1];
            let x = shuffle(x, shuffle_mask);

            // add digits to group
            let x = char_to_digit(x);
            group = _mm256_add_epi8(group, x);
            group_size += 1;

            if group_size == MAX_GROUP_SIZE {
                // parse group
                res += parse_impl(group);

                // reset group
                group = _mm256_setzero_si256();
                group_size = 0;
            }

            // skip processed numbers
            s = &s[d1 + d2 + d3 + 3..];
        }

        if group_size > 0 {
            res += parse_impl(group);
        }

        res + parse_naive_loop(s)
    }
}

fn parse_multithreading<const THREAD_COUNT: usize, F>(s: &[u8], parse: F) -> u64
where
    F: Fn(&[u8]) -> u64 + Send + Sync,
{
    fn split<const N: usize>(s: &[u8]) -> [&[u8]; N] {
        let mut chunks: [MaybeUninit<&[u8]>; N] = unsafe { MaybeUninit::uninit().assume_init() };

        let chunk_size = s.len() / N;
        let mut chunk_start = s;
        for i in 0..N - 1 {
            let mut cur_chunk_size = chunk_size;

            // can't split at the middle of a number, look for next space
            while chunk_start[cur_chunk_size] != SPACE {
                cur_chunk_size += 1;
            }

            chunks[i].write(&chunk_start[..cur_chunk_size]);
            chunk_start = &chunk_start[cur_chunk_size + 1..];
        }
        chunks[N - 1].write(chunk_start);

        unsafe { mem::transmute_copy::<_, [&[u8]; N]>(&chunks) }
    }

    let chunks = split::<THREAD_COUNT>(s);
    let mut res = 0;
    thread::scope(|scope| {
        let threads = chunks.map(|c| scope.spawn(|| parse(c)));
        for t in threads {
            res += t.join().unwrap();
        }
    });

    res
}

fn parse(s: &[u8]) -> u64 {
    unsafe {
        let shuffle_masks = simd256::gen_shuffle_masks();
        parse_multithreading::<8, _>(s, |s| simd256::parse(s, &shuffle_masks))
    }
}

mod mmap {
    #[link(name = "c")]
    extern "C" {
        fn mmap(addr: *mut u8, len: usize, prot: i32, flags: i32, fd: i32, offset: i64) -> *mut u8;
        fn __errno_location() -> *const i32;
        fn lseek(fd: i32, offset: i64, whence: i32) -> i64;
    }

    pub unsafe fn from_stdin<'a>() -> &'a [u8] {
        mmap_fd(0)
    }

    unsafe fn mmap_fd<'a>(fd: i32) -> &'a [u8] {
        let seek_end = 2;
        let size = lseek(fd, 0, seek_end);
        if size == -1 {
            panic!("lseek failed, errno {}", *__errno_location());
        }
        let prot_read = 0x01;
        let map_private = 0x02;
        let map_populate = 0x08000;
        let ptr = mmap(
            0 as _,
            size as usize,
            prot_read,
            map_private | map_populate,
            fd,
            0,
        );
        if ptr as isize == -1 {
            panic!("mmap failed, errno {}", *__errno_location());
        }
        std::slice::from_raw_parts(ptr, size as usize)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    unsafe {
        let input = mmap::from_stdin();
        println!("{}", parse(&input));
    }
    Ok(())
}
