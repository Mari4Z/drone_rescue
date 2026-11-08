import * as THREE from "three";
import type { DroneConfig, EpisodeState } from "./types";

const SCENE_CONFIG: DroneConfig = {
  world: {
    cell_size_m: 5,
    grid_cells: [20, 20, 12],
    base_center_m: [2.5, 2.5, 0],
    base_radius_m: 3.5,
    delivery_center_m: [97.5, 97.5, 0],
    delivery_radius_m: 8,
    obstacles: [
      { min_m: [25, 20, 0], max_m: [35, 30, 30] },
      { min_m: [65, 65, 0], max_m: [75, 75, 30] },
    ],
  },
  physics: { drone_radius_m: 0.5 },
};

export class DroneScene {
  private readonly scene = new THREE.Scene();
  private readonly camera = new THREE.PerspectiveCamera(55, 1, 0.1, 1000);
  private readonly renderer: THREE.WebGLRenderer;
  private readonly drone = new THREE.Group();
  private readonly resizeObserver: ResizeObserver;
  private readonly gridRoot = new THREE.Group();
  private disposed = false;
  private lastFrame = 0;

  constructor(private readonly host: HTMLElement) {
    this.scene.background = new THREE.Color("#0b1210");
    this.camera.position.set(50, 165, 50);
    this.camera.lookAt(50, 0, 50);

    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.12;
    this.renderer.domElement.setAttribute("aria-label", "Vista 3D de la simulación Python");
    this.host.append(this.renderer.domElement);

    this.scene.add(new THREE.HemisphereLight("#c6e8d6", "#17231d", 2.1));
    const sun = new THREE.DirectionalLight("#f0c98e", 2.4);
    sun.position.set(-25, 105, -10);
    sun.castShadow = true;
    sun.shadow.mapSize.set(1024, 1024);
    sun.shadow.camera.left = -80;
    sun.shadow.camera.right = 80;
    sun.shadow.camera.top = 80;
    sun.shadow.camera.bottom = -80;
    this.scene.add(sun);

    this.scene.add(this.gridRoot);
    this.createDrone();
    this.drone.position.set(
      SCENE_CONFIG.world.base_center_m[0],
      1.15,
      SCENE_CONFIG.world.base_center_m[1],
    );
    this.setEnvironment(SCENE_CONFIG);

    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(this.host);
    this.resize();
    this.renderer.setAnimationLoop((time) => this.render(time));
  }

  setEnvironment(config: DroneConfig): void {
    this.clearGroup(this.gridRoot);
    const { world } = config;
    const width = world.grid_cells[0] * world.cell_size_m;
    const depth = world.grid_cells[1] * world.cell_size_m;

    const floor = new THREE.Mesh(
      new THREE.PlaneGeometry(width, depth),
      new THREE.MeshStandardMaterial({ color: "#18231e", roughness: 0.96, metalness: 0.02 }),
    );
    floor.rotation.x = -Math.PI / 2;
    floor.position.set(width / 2, -0.12, depth / 2);
    floor.receiveShadow = true;
    this.gridRoot.add(floor);

    const grid = new THREE.GridHelper(width, world.grid_cells[0], "#557563", "#293b31");
    grid.position.set(width / 2, 0, depth / 2);
    const gridMaterial = grid.material;
    if (Array.isArray(gridMaterial)) {
      gridMaterial.forEach((material) => {
        material.transparent = true;
        material.opacity = 0.24;
      });
    } else {
      gridMaterial.transparent = true;
      gridMaterial.opacity = 0.24;
    }
    this.gridRoot.add(grid);

    for (const obstacle of world.obstacles) {
      const size = new THREE.Vector3(
        obstacle.max_m[0] - obstacle.min_m[0],
        obstacle.max_m[2] - obstacle.min_m[2],
        obstacle.max_m[1] - obstacle.min_m[1],
      );
      const column = new THREE.Mesh(
        new THREE.BoxGeometry(size.x, size.y, size.z),
        new THREE.MeshStandardMaterial({ color: "#b2714e", roughness: 0.78 }),
      );
      column.position.set(
        obstacle.min_m[0] + size.x / 2,
        size.y / 2,
        obstacle.min_m[1] + size.z / 2,
      );
      column.castShadow = true;
      column.receiveShadow = true;
      this.gridRoot.add(column);
      const cap = new THREE.Mesh(
        new THREE.BoxGeometry(size.x * 1.04, 0.34, size.z * 1.04),
        new THREE.MeshStandardMaterial({ color: "#d18c5a", roughness: 0.62 }),
      );
      cap.position.set(column.position.x, size.y, column.position.z);
      this.gridRoot.add(cap);
    }

    this.addZone(world.base_center_m, world.base_radius_m, "#d5b16a", "#57482a");
    this.addZone(
      world.delivery_center_m,
      world.delivery_radius_m,
      "#75c9a4",
      "#244b3a",
    );
  }

  update(state: EpisodeState): void {
    const [x, z, y] = state.position_m;
    this.drone.position.set(x, y + 1.15, z);
  }

  dispose(): void {
    this.disposed = true;
    this.renderer.setAnimationLoop(null);
    this.resizeObserver.disconnect();
    this.renderer.dispose();
    this.scene.traverse((object) => {
      if (object instanceof THREE.Mesh) {
        object.geometry.dispose();
        const materials = Array.isArray(object.material) ? object.material : [object.material];
        materials.forEach((material) => material.dispose());
      }
    });
    this.renderer.domElement.remove();
  }

  private addZone(
    location: [number, number, number],
    radius: number,
    color: string,
    fill: string,
  ): void {
    const [x, z] = location;
    const ring = new THREE.Mesh(
      new THREE.RingGeometry(radius - 0.55, radius, 48),
      new THREE.MeshBasicMaterial({ color, side: THREE.DoubleSide }),
    );
    ring.rotation.x = -Math.PI / 2;
    ring.position.set(x, 0.04, z);
    this.gridRoot.add(ring);
    const disc = new THREE.Mesh(
      new THREE.CircleGeometry(radius - 0.65, 48),
      new THREE.MeshBasicMaterial({ color: fill, transparent: true, opacity: 0.78 }),
    );
    disc.rotation.x = -Math.PI / 2;
    disc.position.set(x, 0.025, z);
    this.gridRoot.add(disc);
    const tag = new THREE.Mesh(
      new THREE.BoxGeometry(4.5, 0.3, 0.3),
      new THREE.MeshBasicMaterial({ color }),
    );
    tag.position.set(x, 0.25, z);
    this.gridRoot.add(tag);
  }

  private createDrone(): void {
    const body = new THREE.Mesh(
      new THREE.BoxGeometry(1.35, 0.36, 1.05),
      new THREE.MeshStandardMaterial({ color: "#dbe8da", metalness: 0.26, roughness: 0.4 }),
    );
    body.castShadow = true;
    body.receiveShadow = true;
    this.drone.add(body);

    const accent = new THREE.Mesh(
      new THREE.BoxGeometry(0.52, 0.22, 0.36),
      new THREE.MeshStandardMaterial({ color: "#ef8f4e", roughness: 0.54 }),
    );
    accent.position.set(0, 0.27, 0);
    this.drone.add(accent);

    const material = new THREE.MeshStandardMaterial({ color: "#bdc9c1", metalness: 0.52, roughness: 0.34 });
    const rotorMaterial = new THREE.MeshStandardMaterial({ color: "#37463e", metalness: 0.2, roughness: 0.55 });
    for (const x of [-0.86, 0.86]) {
      for (const z of [-0.7, 0.7]) {
        const arm = new THREE.Mesh(new THREE.CylinderGeometry(0.055, 0.055, 1.25, 8), material);
        arm.rotation.z = Math.PI / 2;
        arm.position.set(x / 2, 0.02, z / 2);
        this.drone.add(arm);
        const motor = new THREE.Mesh(new THREE.CylinderGeometry(0.14, 0.14, 0.15, 12), material);
        motor.position.set(x, 0.13, z);
        this.drone.add(motor);
        const rotor = new THREE.Mesh(new THREE.BoxGeometry(0.82, 0.035, 0.13), rotorMaterial);
        rotor.position.set(x, 0.24, z);
        this.drone.add(rotor);
      }
    }
    const beacon = new THREE.PointLight("#f68b4f", 2.2, 7);
    beacon.position.set(0, 0.48, 0);
    this.drone.add(beacon);
    this.scene.add(this.drone);
  }

  private clearGroup(group: THREE.Group): void {
    for (const child of [...group.children]) {
      group.remove(child);
      if (child instanceof THREE.Mesh) {
        child.geometry.dispose();
        const materials = Array.isArray(child.material) ? child.material : [child.material];
        materials.forEach((material) => material.dispose());
      }
      if (child instanceof THREE.GridHelper) {
        const materials = Array.isArray(child.material) ? child.material : [child.material];
        materials.forEach((material) => material.dispose());
      }
    }
  }

  private resize(): void {
    const width = this.host.clientWidth;
    const height = this.host.clientHeight;
    if (!width || !height) return;
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height, false);
  }

  private render(time: number): void {
    if (this.disposed) return;
    const delta = Math.min((time - this.lastFrame) / 1000, 0.05);
    this.lastFrame = time;
    this.drone.rotation.y += delta * 0.08;
    this.renderer.render(this.scene, this.camera);
  }
}
