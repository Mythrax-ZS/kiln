use super::*;

#[test]
fn a_superframe_index_splits_a_chunk_into_its_frames() {
    // Two frames of 3 and 2 bytes, sizes on one byte: marker 0b110, 0 (one byte), 1 (two frames).
    let marker = 0b1100_0001u8;
    let chunk = [1, 2, 3, 4, 5, marker, 3, 2, marker];
    let frames = split_superframe(&chunk);
    assert_eq!(frames, vec![&[1u8, 2, 3][..], &[4u8, 5][..]]);
    // Without the index, or with a final byte that is no marker, the chunk is one frame.
    assert_eq!(split_superframe(&[1, 2, 3]), vec![&[1u8, 2, 3][..]]);
    let bad = [1, 2, 3, 4, 5, 0, 3, 2, marker];
    assert_eq!(split_superframe(&bad).len(), 1);
}
