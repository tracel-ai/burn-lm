use burn_lm_inference::Backend;
use burn_std::device::{Device, DeviceId};

pub struct DeviceMapping<D: Device> {
    pub min_layer: u32,
    pub max_layer: u32,
    pub device: D,
}
pub fn map_layers_to_devices<D: Device>(max_layers: u32) -> Vec<DeviceMapping<D>> {
    let total_devices = D::device_count_total() as u32;
    let step = ((max_layers as f32) / (total_devices as f32)).ceil() as u32;

    let all_devices: Vec<_> = (0u32..total_devices)
        .map(|i| {
            let device_id = DeviceId::new(0, i as u32);
            let device = D::from_id(device_id);
            let min = i * step;
            let max = (min + step - 1).min(max_layers - 1);
            DeviceMapping {
                min_layer: min,
                max_layer: max,
                device,
            }
        })
        .collect();
    all_devices
}

impl<D: Device> DeviceMapping<D> {
    pub fn device(&self) -> &D {
        &self.device
    }
    pub fn has_layer(&self, layer: u32) -> bool {
        layer >= self.min_layer && layer <= self.max_layer
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Debug, Default)]
    struct DummyDevice;

    impl burn_std::device::Device for DummyDevice {
        fn from_id(_id: DeviceId) -> Self {
            DummyDevice
        }

        fn device_count_total() -> usize {
            3
        }

        fn to_id(&self) -> DeviceId {
            DeviceId::new(0, 0)
        }

        fn device_count(_type_id: u16) -> usize {
            3
        }
    }

    #[test]
    fn test_assignment() {
        let result = map_layers_to_devices::<DummyDevice>(8);
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].min_layer, 0);
        assert_eq!(result[0].max_layer, 2);

        assert_eq!(result[1].min_layer, 3);
        assert_eq!(result[1].max_layer, 5);

        assert_eq!(result[2].min_layer, 6);
        assert_eq!(result[2].max_layer, 7);
    }
}
