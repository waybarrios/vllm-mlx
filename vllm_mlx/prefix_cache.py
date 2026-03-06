    def _extract_block_tensor_slice(
        self,
        cache_data: List[Dict[str, Any]],
        start_idx: int,
        end_idx: int,
    ) -> Optional[List[Tuple[Any, Any]]]:
        """
        Extract tensor slices for a single block from cache data.

        Args:
            cache_data: List of layer states, each containing 'state': (keys, values)
            start_idx: Start token index in the sequence
            end_idx: End token index in the sequence

        Returns:
            List of (keys_slice, values_slice) for each layer, or None on failure
        """
        if not HAS_MLX or not cache_data:
            return None

        try:
            block_slices = []
            for layer_state in cache_data:
                if "state" not in layer_state:
                    continue

                keys, values = layer_state["state"]

                # KV cache shape varies by model architecture:
                #   4D: (batch, n_kv_heads, seq_len, head_dim) — most models
                #   3D: (n_kv_heads, seq_len, head_dim)        — e.g. Qwen3.5
                # Determine the sequence dimension dynamically
                ndim = keys.ndim if hasattr(keys, "ndim") else len(keys.shape)
                seq_axis = ndim - 2  # seq_len is always second-to-last
                seq_len = keys.shape[seq_axis] if hasattr(keys, "shape") else 0

                if end_idx > seq_len:
                    # Requested range extends beyond available data
                    logger.debug(
                        f"Block slice [{start_idx}:{end_idx}] exceeds seq_len {seq_len}"
                    )
                    # Use whatever is available
                    actual_end = min(end_idx, seq_len)
                    if start_idx >= actual_end:
                        continue
                    # Build a dynamic slice that works for both 3D and 4D tensors
                    slices = tuple(
                        slice(start_idx, actual_end) if i == seq_axis else slice(None)
                        for i in range(ndim)
                    )
                    keys_slice = keys[slices]
                    values_slice = values[slices]
                else:
                    slices = tuple(
                        slice(start_idx, end_idx) if i == seq_axis else slice(None)
                        for i in range(ndim)
                    )
                    keys_slice = keys[slices]
                    values_slice = values[slices]

                block_slices.append((keys_slice, values_slice))

            return block_slices if block_slices else None

        except Exception as e:
            logger.warning(f"Failed to extract block tensor slice: {e}")
            return None