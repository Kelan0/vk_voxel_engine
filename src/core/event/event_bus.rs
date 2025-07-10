use std::any::{Any, TypeId};
use std::collections::HashMap;
use log::error;
use shrev::{EventChannel, EventIterator, ReaderId};

pub use shrev::Event as Event;

pub struct EventBus {
    event_channels: HashMap<TypeId, Box<dyn Any>>
}

impl EventBus {
    pub fn new() -> EventBus {
        EventBus{
            event_channels: HashMap::new(),
        }
    }
    
    pub fn channel<E: Event>(&mut self) -> &mut EventChannel<E> {
        let type_id = TypeId::of::<E>();
        
        if !self.event_channels.contains_key(&type_id) {
            self.event_channels.insert(type_id, Box::new(EventChannel::<E>::new()));
        }
        
        self.event_channels
            .get_mut(&type_id)
            .and_then(|boxed| boxed.downcast_mut::<EventChannel<E>>())
            .expect("EventChannel downcast failed")
    }
    
    pub fn emit<E: Event>(&mut self, event: E) {
        let channel = self.channel::<E>();
        channel.single_write(event);
    }
    
    pub fn register<E: Event>(&mut self) -> ReaderId<E> {
        let channel = self.channel::<E>();
        channel.register_reader()
    }
    
    pub fn read<E: Event>(&mut self, reader_id: &mut ReaderId<E>) -> EventIterator<E> {
        let channel = self.channel::<E>();
        channel.read(reader_id)
    }

    pub fn read_opt<E: Event>(&mut self, reader_id: &mut Option<ReaderId<E>>) -> EventIterator<E> {
        if cfg!(debug_assertions) {
            if reader_id.is_none() {
                error!("Tried to read EventBus for Option reader_id, but reader_id is None");
            }
        }
        let r = reader_id.as_mut().unwrap();
        self.read(r)
    }

    pub fn read_one<E: Event>(&mut self, reader_id: &mut ReaderId<E>) -> Option<&E> {
        self.read(reader_id).next()
    }

    pub fn read_one_opt<E: Event>(&mut self, reader_id: &mut Option<ReaderId<E>>) -> Option<&E> {
        self.read_opt(reader_id).next()
    }

    pub fn has_any<E: Event>(&mut self, reader_id: &mut ReaderId<E>) -> bool {
        self.read_one(reader_id).is_some()
    }

    pub fn has_any_opt<E: Event>(&mut self, reader_id: &mut Option<ReaderId<E>>) -> bool {
        self.read_one_opt(reader_id).is_some()
    }
}
