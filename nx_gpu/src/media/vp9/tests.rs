use super::*;

#[test]
fn a_written_key_frame_decodes_to_gray_with_its_color_space() {
    let color = Color {
        space: CS_BT_709,
        full_range: true,
    };
    let mut d = Decoder::new();
    let frames = d.decode(&gray_key_frame(color)).unwrap();
    assert_eq!(frames.len(), 1, "the frame is shown");
    let f = &frames[0];
    assert_eq!((f.width, f.height), (8, 8));
    for (i, p) in f.planes.iter().enumerate() {
        assert!(p.visible().iter().all(|&s| s == 128), "plane {i}");
    }
    assert_eq!(d.color(), color);
    let mut d = Decoder::new();
    d.decode(&gray_key_frame(Color::default())).unwrap();
    assert_eq!(d.color(), Color::default());
}

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
