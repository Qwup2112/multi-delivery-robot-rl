/*
 * Cay nhi phan tim kiem (BST)
 *
 * Quy uoc: node co ten `root` la node goc cua cay.
 *
 *  a) tim_node_me, tim_lien_truoc, tim_lien_sau
 *  b) xoa_node      (3 truong hop: node la, node 1 con, node 2 con)
 *  c) thay_doi_khoa
 *  d) main: kiem tra tat ca cac ham tren
 *
 * Bien dich:  gcc -Wall -Wextra -std=c99 bst.c -o bst && ./bst
 */
#include <stdio.h>
#include <stdlib.h>

struct Node {
    int key;
    struct Node *left;
    struct Node *right;
};

/* ===================== Cac ham co ban ===================== */

struct Node *tao_node(int key)
{
    struct Node *n = (struct Node *)malloc(sizeof(struct Node));
    if (n == NULL) {
        fprintf(stderr, "Khong du bo nho\n");
        exit(1);
    }
    n->key = key;
    n->left = NULL;
    n->right = NULL;
    return n;
}

/* Chen khoa vao cay, tra ve goc (moi) cua cay. Khoa trung bi bo qua. */
struct Node *chen(struct Node *root, int key)
{
    if (root == NULL)
        return tao_node(key);
    if (key < root->key)
        root->left = chen(root->left, key);
    else if (key > root->key)
        root->right = chen(root->right, key);
    return root;
}

struct Node *tim_kiem(struct Node *root, int key)
{
    while (root != NULL && root->key != key)
        root = (key < root->key) ? root->left : root->right;
    return root;
}

struct Node *tim_min(struct Node *root)
{
    if (root == NULL)
        return NULL;
    while (root->left != NULL)
        root = root->left;
    return root;
}

struct Node *tim_max(struct Node *root)
{
    if (root == NULL)
        return NULL;
    while (root->right != NULL)
        root = root->right;
    return root;
}

void duyet_giua(struct Node *root)
{
    if (root == NULL)
        return;
    duyet_giua(root->left);
    printf("%d ", root->key);
    duyet_giua(root->right);
}

/* In cay xoay ngang 90 do (cay con phai o tren, cay con trai o duoi) */
void in_cay(struct Node *root, int muc)
{
    if (root == NULL)
        return;
    in_cay(root->right, muc + 1);
    for (int i = 0; i < muc; i++)
        printf("      ");
    printf("%d\n", root->key);
    in_cay(root->left, muc + 1);
}

void giai_phong(struct Node *root)
{
    if (root == NULL)
        return;
    giai_phong(root->left);
    giai_phong(root->right);
    free(root);
}

/* Kiem tra tinh chat BST: moi khoa nam trong khoang (lo, hi) */
static int la_bst(struct Node *root, long lo, long hi)
{
    if (root == NULL)
        return 1;
    if (root->key <= lo || root->key >= hi)
        return 0;
    return la_bst(root->left, lo, root->key) && la_bst(root->right, root->key, hi);
}

/* ===================== a) Cac ham phu tro ===================== */

/*
 * Tim node me cua node_cho_truoc.
 * Tra ve NULL neu node_cho_truoc la goc, la NULL, hoac khong nam trong cay.
 */
struct Node *tim_node_me(struct Node *root, struct Node *node_cho_truoc)
{
    if (root == NULL || node_cho_truoc == NULL || root == node_cho_truoc)
        return NULL;

    struct Node *me = NULL;
    struct Node *cur = root;
    int key = node_cho_truoc->key;

    while (cur != NULL && cur != node_cho_truoc) {
        me = cur;
        cur = (key < cur->key) ? cur->left : cur->right;
    }
    /* cur == NULL nghia la node_cho_truoc khong thuoc cay */
    return (cur == node_cho_truoc) ? me : NULL;
}

/*
 * Tim node lien truoc (khoa lon nhat nho hon khoa cua node_cho_truoc).
 *  - Neu co cay con trai: la node lon nhat cua cay con trai.
 *  - Neu khong: la to tien gan nhat ma node_cho_truoc nam o cay con PHAI cua no.
 * Tra ve NULL neu khong co.
 */
struct Node *tim_lien_truoc(struct Node *root, struct Node *node_cho_truoc)
{
    if (root == NULL || node_cho_truoc == NULL)
        return NULL;

    if (node_cho_truoc->left != NULL)
        return tim_max(node_cho_truoc->left);

    struct Node *lien_truoc = NULL;
    struct Node *cur = root;
    int key = node_cho_truoc->key;

    while (cur != NULL && cur != node_cho_truoc) {
        if (key > cur->key) {
            lien_truoc = cur;   /* re phai: cur la ung vien */
            cur = cur->right;
        } else {
            cur = cur->left;
        }
    }
    return (cur == node_cho_truoc) ? lien_truoc : NULL;
}

/*
 * Tim node lien sau (khoa nho nhat lon hon khoa cua node_cho_truoc).
 *  - Neu co cay con phai: la node nho nhat cua cay con phai.
 *  - Neu khong: la to tien gan nhat ma node_cho_truoc nam o cay con TRAI cua no.
 * Tra ve NULL neu khong co.
 */
struct Node *tim_lien_sau(struct Node *root, struct Node *node_cho_truoc)
{
    if (root == NULL || node_cho_truoc == NULL)
        return NULL;

    if (node_cho_truoc->right != NULL)
        return tim_min(node_cho_truoc->right);

    struct Node *lien_sau = NULL;
    struct Node *cur = root;
    int key = node_cho_truoc->key;

    while (cur != NULL && cur != node_cho_truoc) {
        if (key < cur->key) {
            lien_sau = cur;     /* re trai: cur la ung vien */
            cur = cur->left;
        } else {
            cur = cur->right;
        }
    }
    return (cur == node_cho_truoc) ? lien_sau : NULL;
}

/* ===================== b) Xoa node ===================== */

/*
 * Xoa node co toi da 1 con (truong hop 1 va 2).
 * Vi `root` duoc truyen theo gia tri, khong the doi con tro goc cua nguoi goi,
 * nen khi node can xoa chinh la goc va co 1 con, ta chep noi dung node con
 * len goc roi giai phong node con.
 */
static int xoa_node_toi_da_1_con(struct Node *root, struct Node *node)
{
    struct Node *con = (node->left != NULL) ? node->left : node->right;

    if (node == root) {
        if (con == NULL)
            return 0;   /* cay chi co 1 node: khong the xoa goc qua con tro truyen gia tri */
        root->key = con->key;
        root->left = con->left;
        root->right = con->right;
        free(con);
        return 1;
    }

    struct Node *me = tim_node_me(root, node);
    if (me == NULL)
        return 0;
    if (me->left == node)
        me->left = con;
    else
        me->right = con;
    free(node);
    return 1;
}

/*
 * Xoa node_can_xoa khoi cay. Tra ve 1 neu thanh cong, 0 neu that bai.
 *  (1) Node la      : bo lien ket tu node me, giai phong node.
 *  (2) Node 1 con   : noi node me truc tiep voi node con duy nhat.
 *  (3) Node 2 con   : chep khoa cua node lien sau (min cay con phai) vao
 *                     node_can_xoa, roi xoa node lien sau (no co toi da 1 con).
 */
int xoa_node(struct Node *root, struct Node *node_can_xoa)
{
    if (root == NULL || node_can_xoa == NULL)
        return 0;
    /* Node phai thuoc cay */
    if (node_can_xoa != root && tim_node_me(root, node_can_xoa) == NULL)
        return 0;

    if (node_can_xoa->left != NULL && node_can_xoa->right != NULL) {
        struct Node *lien_sau = tim_lien_sau(root, node_can_xoa);
        node_can_xoa->key = lien_sau->key;
        return xoa_node_toi_da_1_con(root, lien_sau);
    }
    return xoa_node_toi_da_1_con(root, node_can_xoa);
}

/* ===================== c) Thay doi khoa ===================== */

/*
 * Doi khoa cua node co khoa `khoa_cu` thanh `khoa_moi`.
 * Khong the gan truc tiep vi se pha vo tinh chat BST, nen ta xoa node cu
 * roi chen khoa moi vao dung vi tri.
 * That bai (tra ve 0) neu: khong co khoa_cu, hoac khoa_moi da ton tai.
 */
int thay_doi_khoa(struct Node *root, const int khoa_cu, const int khoa_moi)
{
    struct Node *node = tim_kiem(root, khoa_cu);
    if (node == NULL)
        return 0;
    if (khoa_cu == khoa_moi)
        return 1;
    if (tim_kiem(root, khoa_moi) != NULL)
        return 0;

    /* Cay chi co 1 node: doi truc tiep */
    if (root->left == NULL && root->right == NULL) {
        root->key = khoa_moi;
        return 1;
    }

    if (!xoa_node(root, node))
        return 0;
    chen(root, khoa_moi);   /* root != NULL nen goc khong doi */
    return 1;
}

/* ===================== d) main ===================== */

static void in_ket_qua_node(const char *ten, int key, struct Node *kq)
{
    if (kq != NULL)
        printf("  %s(%d) = %d\n", ten, key, kq->key);
    else
        printf("  %s(%d) = NULL\n", ten, key);
}

static void hien_thi(struct Node *root)
{
    printf("  Duyet giua: ");
    duyet_giua(root);
    printf("\n  BST hop le: %s\n", la_bst(root, -2147483649L, 2147483648L) ? "co" : "KHONG");
    in_cay(root, 1);
}

int main(void)
{
    /*
     * Cay BST ban dau:
     *
     *                 50
     *              /      \
     *            30        70
     *           /  \      /  \
     *         20    40  60    80
     *        /     /  \   \
     *      10    35   45   65
     */
    int keys[] = {50, 30, 70, 20, 40, 60, 80, 10, 35, 45, 65};
    int n = sizeof(keys) / sizeof(keys[0]);
    struct Node *root = NULL;
    for (int i = 0; i < n; i++)
        root = chen(root, keys[i]);

    printf("=== Cay ban dau ===\n");
    hien_thi(root);

    /* ---------- a) Node me, lien truoc, lien sau ---------- */
    printf("\n=== a) Tim node me / lien truoc / lien sau ===\n");
    int test_keys[] = {50, 30, 10, 45, 65, 80, 35, 60};
    for (unsigned i = 0; i < sizeof(test_keys) / sizeof(test_keys[0]); i++) {
        struct Node *x = tim_kiem(root, test_keys[i]);
        printf("Node %d:\n", x->key);
        in_ket_qua_node("tim_node_me   ", x->key, tim_node_me(root, x));
        in_ket_qua_node("tim_lien_truoc", x->key, tim_lien_truoc(root, x));
        in_ket_qua_node("tim_lien_sau  ", x->key, tim_lien_sau(root, x));
    }

    /* ---------- b) Xoa node ---------- */
    printf("\n=== b) Xoa node ===\n");

    printf("\n(1) Xoa node la 10: %d\n", xoa_node(root, tim_kiem(root, 10)));
    hien_thi(root);

    printf("\n(2) Xoa node 1 con 60 (con phai 65): %d\n", xoa_node(root, tim_kiem(root, 60)));
    hien_thi(root);

    printf("\n(3) Xoa node 2 con 30: %d\n", xoa_node(root, tim_kiem(root, 30)));
    hien_thi(root);

    printf("\n(3) Xoa node goc 50 (2 con): %d\n", xoa_node(root, tim_kiem(root, 50)));
    hien_thi(root);

    printf("\nXoa node khong ton tai (NULL): %d (mong doi 0)\n", xoa_node(root, tim_kiem(root, 999)));
    struct Node *ngoai_cay = tao_node(40);
    printf("Xoa node khong thuoc cay: %d (mong doi 0)\n", xoa_node(root, ngoai_cay));
    free(ngoai_cay);

    /* ---------- c) Thay doi khoa ---------- */
    printf("\n=== c) Thay doi khoa ===\n");

    printf("\nDoi 45 -> 75: %d\n", thay_doi_khoa(root, 45, 75));
    hien_thi(root);

    printf("\nDoi 80 -> 5: %d\n", thay_doi_khoa(root, 80, 5));
    hien_thi(root);

    int khoa_goc = root->key;
    printf("\nDoi goc %d -> 42: %d\n", khoa_goc, thay_doi_khoa(root, khoa_goc, 42));
    hien_thi(root);

    printf("\nDoi 999 -> 1 (khoa cu khong ton tai): %d (mong doi 0)\n", thay_doi_khoa(root, 999, 1));
    printf("Doi 35 -> 70 (khoa moi da ton tai): %d (mong doi 0)\n", thay_doi_khoa(root, 35, 70));
    hien_thi(root);

    giai_phong(root);
    return 0;
}
